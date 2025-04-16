import torch
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import apply_config, CustomLLMPagConfig, LLMPagConfig
from data.data_module import LMDataModule
from data.data_processor import BatchType
from infer_tinystories import load_model
from models.masked_embeddings_grad_model import MaskedIdentityGradEmbeddingsModel


def sim_matrix(a, b, eps=1e-8):
    """
    torch.cdist, but with cosine similarity

    Source: https://stackoverflow.com/a/67588366
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@apply_config('tiny-train')
@torch.no_grad()
def main(cfg: CustomLLMPagConfig | LLMPagConfig):
    lightning_model = load_model(cfg)
    assert isinstance(lightning_model, MaskedIdentityGradEmbeddingsModel), \
        f'Model is not MaskedIdentityGradEmbeddingsModel, but {type(lightning_model)}'

    assert lightning_model.tokenizer.padding_side == 'right', \
        f'Expected tokenizer padding side to be right, but got {lightning_model.tokenizer.padding_side}'

    v, d = lightning_model.tokenizer.vocab_size, lightning_model.model.config.hidden_size
    assert v == 2048, \
        f'Expected vocab size to be 2048, but got {v}'
    assert d == 128, \
        f'Expected hidden size to be 128, but got {d}'

    top_k_for_accuracy = 200

    # Create data module
    data_module = LMDataModule(cfg, lightning_model.tokenizer)
    data_module.prepare_data()
    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    all_embeddings = lightning_model.model.get_input_embeddings().weight
    assert all_embeddings.shape == (v, d), \
        f'Expected all_embeddings shape {(v, d)}, but got {all_embeddings.shape}'

    accuracy = torch.tensor(0, device=lightning_model.device, dtype=torch.long)

    # We want to classify the first token, given the gradients on the embedding layer
    for batch in tqdm(train_dataloader):
        batch: BatchType
        batch.to(lightning_model.device)
        x_input = batch.input_ids
        n = x_input.size(0)
        k = 10

        # Take the last 30 tokens of the x_input, per item
        # Remember that the padding is on the left
        x_lengths = batch.attention_mask.sum(dim=1)
        assert x_lengths.shape == (n,), \
            f'Expected x_lengths shape {n}, but got {x_lengths.shape}'

        # x_last_token = x_lengths - 1
        # assert torch.equal(batch.attention_mask[:, x_last_token][torch.eye(n) == 1], torch.ones(n)), \
        #     'Something is wrong about the last token index'

        x_ids_partial = x_input[:, k:].clone()
        x_attn_partial = batch.attention_mask[:, k:]
        assert torch.equal(
            batch.attention_mask[:, k],
            torch.ones(n, dtype=torch.long, device=lightning_model.device),
        )

        # We want to predict the first token
        y_true = x_ids_partial[:, 0].clone()

        # Hide the first token from the input
        x_ids_partial[:, 0] = lightning_model.tokenizer.pad_token_id

        # Do a forward pass
        x_embeds_partial = lightning_model.model.get_input_embeddings()(x_ids_partial)
        with torch.set_grad_enabled(True):
            x_embeds_partial.requires_grad_(True)
            masked_outputs: CausalLMOutputWithPast = lightning_model.model(inputs_embeds=x_embeds_partial,
                                                                           attention_mask=x_attn_partial,
                                                                           labels=x_ids_partial,  # FIXME
                                                                           output_hidden_states=False)
            x_grads = torch.autograd.grad(
                masked_outputs.loss,
                [x_embeds_partial],
                create_graph=False,
            )[0]

        # Calculate pairwise L2 distances using torch.cdist
        # The result `dist_matrix` will have shape (N, V)
        # where dist_matrix[i, j] is the L2 distance between N_tensor[i] and M_tensor[j]
        y_hat = x_grads[:, 0, :]
        dist_matrix = torch.cdist(y_hat, all_embeddings, p=2)
        assert dist_matrix.shape == (n, v), \
            f'Expected dist_matrix shape {(n, v)}, but got {dist_matrix.shape}'

        # Get the prediction for each sample
        y_pred = torch.argmin(dist_matrix, dim=1)

        assert y_pred.shape == (n,), \
            f'Expected y_pred shape {(n,)}, but got {y_pred.shape}'

        # accuracy += torch.sum(y_pred == y_true)
        #
        # print('True distances:', dist_matrix[:, y_true][torch.eye(n) == 1])
        # print('Predicted distances:', dist_matrix[:, y_pred][torch.eye(n) == 1])

        sorted_rank_of_y_pred = torch.zeros(n, dtype=torch.long, device=lightning_model.device)
        for i in range(n):  # FIXME: do not use a for loop
            predictions = dist_matrix[i]
            assert predictions.shape == (v,), \
                f'Expected predictions shape {(v,)}, but got {predictions.shape}'

            sorted_predictions, _ = torch.sort(predictions)
            k_pred = torch.eq(sorted_predictions, predictions[y_true[i]]).int().argmax()
            sorted_rank_of_y_pred[i] = k_pred

        # print('It would be K-th prediction:', sorted_rank_of_y_pred)
        accuracy += (sorted_rank_of_y_pred < top_k_for_accuracy).sum()

    tqdm.write(
        f'Final accuracy: {accuracy} / {len(train_dataloader.dataset)} = {accuracy / len(train_dataloader.dataset):.2%}')


if __name__ == "__main__":
    main()
