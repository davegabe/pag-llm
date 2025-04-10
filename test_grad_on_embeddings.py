import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("raincandy-u/TinyStories-656K")
model = AutoModelForCausalLM.from_pretrained("raincandy-u/TinyStories-656K", attn_implementation='eager')
model.eval()

# Divide word embeddings weight tying with LM Head
model.lm_head = torch.nn.Linear(128, 2048, bias=True)

# Create a dummy input
input_ids = torch.tensor([
    [1, 2, 30, 4, 5],
    [1, 2,  3, 4, 5],
])
n, m = input_ids.shape
print('With input_ids:')
print(input_ids)
print()

# Create a dummy attention mask
attn_mask = torch.ones_like(input_ids)

# Create the labels as rolling input_ids
# labels = torch.roll(input_ids, shifts=-1, dims=1)
# labels[:, -1] = 8  # Set the last token
labels = torch.ones_like(input_ids) * 10  # Set all labels to 10
print('With labels:')
print(labels)
print()

# Create the embeddings
embeddings = model.get_input_embeddings()(input_ids)
assert embeddings.shape == (n, m, 128), "Embeddings shape is incorrect"
embeddings.requires_grad_(True)
embeddings.retain_grad()  # Retain gradients for the embeddings

# Forward pass, step by step
outputs = model(inputs_embeds=embeddings,
                attention_mask=attn_mask,
                labels='dummy',  # See https://github.com/huggingface/transformers/issues/32944#issuecomment-2781021624
                shift_labels=labels,
                max_new_tokens=1,
                use_cache=False,
                output_hidden_states=True)

for hidden_state in outputs.hidden_states:
    hidden_state.retain_grad()
outputs.logits.retain_grad()

# Take only the first logits
# take_m = 4
# outputs.logits = outputs.logits[:, :take_m, :]
# labels = labels[:, :take_m]
# loss = F.cross_entropy(outputs.logits.reshape(-1, 2048), labels.reshape(-1))
loss = outputs.loss
loss.backward()

assert model.model.embed_tokens.weight.grad is not None
assert model.lm_head.weight.grad is not None
# assert torch.all(model.lm_head.weight == model.model.embed_tokens.weight), "Weight tying is not set up correctly"
# assert torch.all(model.lm_head.weight.grad == model.model.embed_tokens.weight.grad), "Weight tying is not set up correctly"

print('=== Grads ===')
print('On logits:')
print(outputs.logits.grad.sum(dim=-1))
print()
for layer_i, hidden_state in reversed(list(enumerate(outputs.hidden_states))):
    print(f'On layer {layer_i}:', end=' ')
    if layer_i == 0:
        print('Embeddings')
    elif layer_i == len(outputs.hidden_states) - 1:
        print('LM Head, right before classification')
    else:
        print()
    print(hidden_state.grad.sum(dim=-1))
    print()
