import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import Config
from data.data_processor import BatchType, TextDataset
from models.base_model import BaseLMModel
from utils.index_token_to_dataset_item import DatasetIndexByToken


class PAGHiddenModel(BaseLMModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
            dataset_index: DatasetIndexByToken,
            train_dataset: TextDataset,
    ):
        super().__init__(model, tokenizer, config)
        self.dataset_index = dataset_index.to(model.device)
        self.train_dataset = train_dataset
        self.hidden_layer_index = config.model.hidden_layer_index
        self.pag_classes = config.training.pag_classes # Number of different next tokens to consider
        self.pag_samples = config.training.pag_samples  # Number of different samples for each next token considered
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag

    
    def training_step(self, batch: BatchType, batch_idx: int):
        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(**batch.to_dict(), output_hidden_states=True)
        loss_ce = outputs.loss
        hidden_states = outputs.hidden_states[self.hidden_layer_index]
        hidden_states.requires_grad_(True)

        # Get samples for the next tokens
        #
        # pag_input_ids: [k * m, T]
        # pag_attn_mask: [k * m, T]
        # pag_classes:   [k]
        k, m = self.pag_classes, self.pag_samples
        pag_input_ids, pag_attn_mask, pag_classes = self.dataset_index.get_rand_samples_by_token(
            dataset=self.train_dataset,
            k_classes=k,
            num_samples=m,
        )

        pag_input_ids, pag_attn_mask, pag_classes = \
            map(lambda t: t.to(self.model.device), (pag_input_ids, pag_attn_mask, pag_classes))

        # Get, in a single shot, all the hidden states
        with torch.no_grad():
            pag_output = self.model.generate(
                input_ids=pag_input_ids,
                attention_mask=pag_attn_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=1,  # We only need the hidden states for the prefix tokens,
                    # not actual autoregressive generation
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.train_dataset.tokenizer.pad_token_id,
                )
            )

            # We take [:, -1] since the padding is on the left, at the beginning
            assert pag_attn_mask[:, -1].sum() == (k * m), 'Expected to have padding on the left, not on the right!'
            pag_hidden_states = pag_output.hidden_states[0][self.hidden_layer_index][:, -1].view(k, m, -1)


        # For each next token (classes of the PAG loss)
        loss_cos = torch.tensor(0.0, device=self.device)
        # Iterate for each pag_class
        for class_hidden_layer, ground_truth_token in zip(pag_hidden_states, pag_classes):
            print(f'class_hidden_layer: {class_hidden_layer.shape}')
            print(f'ground_truth_token: {ground_truth_token.shape}')

            # Compute x-x'
            g_batch = hidden_states - class_hidden_layer
            g_batch_y = ground_truth_token * torch.ones_like(g_batch[:, :, 0],
                                                             dtype=torch.long)  # The right token is only one

            # Compute the loss of the batches, with respect to this class
            print(f'{g_batch.shape=}, {g_batch_y.shape=}, {outputs.logits.shape=}')
            vocab_size = outputs.logits.size(-1)
            loss_fixed_class = self.criterion(
                input=outputs.logits.view(-1, vocab_size),
                target=g_batch_y.view(-1),
            )
            batch_z_grad_fixed_class = torch.autograd.grad(
                loss_fixed_class,
                [hidden_states],
                create_graph=True
            )[0]
            
            # Compute the L_cos between grads of batch_z and g_y(batch_z)
            class_loss_cos = F.cosine_similarity(batch_z_grad_fixed_class, g_batch)
            class_loss_cos = 1 - torch.mean(class_loss_cos)

            # Accumulate the loss
            loss_cos += class_loss_cos

        # Average the cosine similarity loss
        loss_cos /= pag_classes.size(0)

        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_cos

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
