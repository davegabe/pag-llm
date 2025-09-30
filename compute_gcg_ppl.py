import json
import re
import sys

import evaluate
from transformers import AutoTokenizer


ILM_TOKENIZER_NAME = 'DaveGabe/TinyStoriesV2_cleaned-voc2048-seq256-overlap25-bpe-tokenizer'
EXTERNAL_LLM_NAME = 'meta-llama/Llama-3.2-1B'

ilm_tokenizer = AutoTokenizer.from_pretrained(ILM_TOKENIZER_NAME)


def clean_text(text: str) -> str:
    """
    Clean the input text by removing special tokens and normalizing whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Remove all special tokens from attack texts
    for tok in ilm_tokenizer.all_special_tokens + ["<|endoftext|>"]:
        text = text.replace(tok, '')
    # Remove all the whitespaces since we have metaspace
    text = re.sub(r'\s+', '', text)
    # Replace metaspace with regular space
    text = text.replace("\u2581", " ").strip()
    return text


def load_sentences(filename: str) -> list[tuple[str, str]]:
    with open(filename) as f:
        data = json.load(f)

    original_prefix_tokens = (
            d['original_prefix_ids']
            for d in data
    )

    original_prefix_strings = [
            clean_text(ilm_tokenizer.decode(tokens, skip_special_tokens=True))
            for tokens in original_prefix_tokens
    ]

    x_attack_strings = [
            clean_text(d['x_attack_str'])
            for d in data
    ]

    return list(zip(original_prefix_strings, x_attack_strings))

def print_example_sentences(sentences: list[str], tag: str, num_examples: int = 5):
    print(f'Example {tag} sentences:')
    print('\n'.join(sentences[:num_examples]))
    print()


def print_sentences_ppl(sentences: list[str], tag: str):
    print(f'\n--- {tag.upper()} Perplexity Results ---')
    print_example_sentences(sentences, tag)

    results = perplexity.compute(
        model_id=EXTERNAL_LLM_NAME,
        predictions=sentences,
        add_start_token=True,
    )
    mean_ppl = results['mean_perplexity']
    stddev_ppl = sum((p - mean_ppl) ** 2 for p in results['perplexities']) / len(results['perplexities']) ** 0.5
    print(f"Mean {tag} perplexity: {results['mean_perplexity']:.2f} ± {stddev_ppl:.2f}")


def print_file_ppl(filename: str):
    print(f'Loading sentences from {filename}...')
    all_sentences = load_sentences(filename)
    print(f'Loaded {len(all_sentences)} sentence pairs.')

    original_sentences = [ original_x for original_x, _ in all_sentences ]
    print_sentences_ppl(original_sentences, 'original')

    attack_sentences = [ attack_x for _, attack_x in all_sentences ]
    print_sentences_ppl(attack_sentences, 'attack')
    print('--- End of Results ---\n')


def main():
    # Define a list of sentences to evaluate
    if len(sys.argv) < 2:
        print(f'USAGE: {sys.argv[0]} GCG_FILE...')
        print()
        exit()

    for filename in sys.argv[1:]:
        print_file_ppl(filename)


# Load the perplexity metric from the Hugging Face hub
try:
    perplexity = evaluate.load("perplexity", module_type="metric")
except Exception as e:
    print(f"Error loading perplexity metric: {e}")
    print("Please make sure you have an internet connection and the necessary libraries are installed.")
    exit()


if __name__ == "__main__":
    main()
