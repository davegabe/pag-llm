import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from base_model import BaseLMModel
from config import Config
from data.data_processor import BatchType
from utils.hdf5 import get_hidden_states_by_next_token, get_all_next_tokens


class PAGHiddenModel(BaseLMModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        hdf5_file_path: str
    ):
        super().__init__(model, tokenizer, config)
        self.hdf5_file_path = hdf5_file_path
        self.used_next_tokens = get_all_next_tokens(hdf5_file_path).to(model.device)
        self.hidden_layer_index = config.model.hidden_layer_index
        self.pag_classes = config.training.pag_classes # Number of different next tokens to consider
        self.criterion = torch.nn.CrossEntropyLoss()


    def get_pag_samples(self) -> torch.Tensor:
        """
        Get hidden states for the next tokens to use for the PAG loss.

        Returns:
            torch.Tensor: Hidden states for the next tokens. Shape: [pag_classes, N, T, D]
        """
        emb_by_next_token = []

        # Get random next tokens
        random_next_tokens = torch.randperm(len(self.used_next_tokens), device=self.device)[:self.pag_classes]
        next_tokens = self.used_next_tokens[random_next_tokens]
        
        # Get hidden states for the next token
        # FIXME: Do not use LOOPS
        for next_token in next_tokens:
            hidden_states, _, next_tokens = get_hidden_states_by_next_token(
                self.hdf5_file_path,
                next_token,
                max_samples=1,
                randomize=True
            )
            print(next_tokens, hidden_states.shape)
            raise Exception()
            emb_by_next_token.append(torch.from_numpy(hidden_states).to(self.device))

        return torch.cat(emb_by_next_token, dim=0)

    
    def training_step(self, batch: BatchType, batch_idx: int):
        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(**batch, output_hidden_states=True)
        loss = outputs.loss
        hidden_states = outputs.hidden_states[self.hidden_layer_index]
        
        # Retrieve next tokens
        next_tokens = batch["labels"]   # [N, T]
        print('next_tokens:', next_tokens.shape)

        # Get hidden states for the next tokens
        embs_by_next_token = self.get_pag_samples()
        print('embs_by_next_token:', embs_by_next_token.shape)

        # For each next token (classes of the PAG loss)
        loss_cos = torch.tensor(0.0, device=self.device)
        for embs in embs_by_next_token:
            # Compute x-x'
            print(hidden_states.shape, embs.shape)
            g_batch = hidden_states - embs
            g_batch_y = next_tokens * torch.ones_like(g_batch)

            # Compute the loss of the batches, with respect to this class
            loss_fixed_class = self.criterion(outputs.logits, g_batch_y)
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

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
