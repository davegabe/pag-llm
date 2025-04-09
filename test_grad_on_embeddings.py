import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("raincandy-u/TinyStories-656K")
model = AutoModelForCausalLM.from_pretrained("raincandy-u/TinyStories-656K")
model.eval()

# Remove weight tying
# assert model.model.embed_tokens.weight.data is model.lm_head.weight.data, "Weight tying is not set up correctly"
# model.model.embed_tokens.weight.data = model.lm_head.weight.data.clone()
# assert model.model.embed_tokens.weight.data is not model.lm_head.weight.data, "Weight tying is not removed correctly"

# Create a dummy input
input_ids = torch.tensor([
    [1, 2, 30, 4, 5, 6, 7],
])
print('With input_ids:', input_ids[0])

assert input_ids.shape == (1, 7), "Input shape is incorrect"

# Create a dummy attention mask
attn_mask = torch.ones_like(input_ids)

# Create the labels as rolling input_ids
labels = torch.roll(input_ids, shifts=-1, dims=1)
labels[:, -1] = 8  # Set the last token
labels = torch.ones_like(input_ids) * 8  # Set all labels to 8
print('With labels:', labels[0])
assert labels.shape == (1, 7), "Labels shape is incorrect"

# Create the embeddings
embeddings = model.get_input_embeddings()(input_ids)
assert embeddings.shape == (1, 7, 128), "Embeddings shape is incorrect"
embeddings.requires_grad_(True)
embeddings.retain_grad()  # Retain gradients for the embeddings

# Forward pass
outputs = model(inputs_embeds=embeddings, attention_mask=attn_mask, labels=labels, max_new_tokens=1)
# loss = outputs.loss
# loss.backward()
outputs.logits[0, 3].sum().backward()  # Backpropagate through the logits

print('We have grad_embeddings:')
print(embeddings.grad.sum(dim=-1))
