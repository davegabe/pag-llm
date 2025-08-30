import json

def print_stats(filename: str, max_samples: int = 10):
    with open(filename) as f:
        samples = json.load(f)['samples'][:max_samples]

    for s in samples:
        y = s['suffix_text']
        y_trimmed = ' '.join(y.split()[:25])

        original_x = s['original_prefix_text']
        original_x = original_x[:original_x.find(y[:50])].replace('\n', ' ')

        predicted_x = s['predicted_prefix_text']
        predicted_x = predicted_x[:predicted_x.find(y[:50])].replace('\n', ' ')

        print(f'\033[96;1mOriginal:  \033[22m{original_x}\033[0m')
        print(f'\033[92;1mPredicted: \033[22m{predicted_x}\033[0m')
        print(y_trimmed, '[...]')
        print()


def _is_notebook() -> bool:
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False


if __name__ == '__main__':
    if _is_notebook():
        print('You can use print_stats now!')
    else:
        print_stats('backward_inference-base-bigram.json')

