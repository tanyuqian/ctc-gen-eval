import argparse

from ctc_score import StyleTransferScorer, SummarizationScorer, DialogScorer
from ctc_score.configs import TASKS, ALIGNS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--align', required=True, choices=ALIGNS,
                        help='the align model to use')
    parser.add_argument('--aspect', required=True,
                        help='the aspect to evaluate')
    parser.add_argument('--hypo', required=True,
                        help='a file with all hypothesized texts to evaluate, '
                             'line-by-line')
    parser.add_argument('--remove_stopwords',
                        default=False, action='store_true',
                        help='whether to remove stopwords in aligning')
    parser.add_argument('--scores_save_path', default='/tmp/scores.txt',
                        help='a path to save example-wise scores.')

    # Dialog
    parser.add_argument('--fact', help='a file with all facts, line-by-line')
    parser.add_argument('--dialog_history',
                        help='a file with all dialog histories, line-by-line')

    # Summarization
    parser.add_argument('--doc', help='a file with all documents, line-by-line')
    parser.add_argument(
        '--refs',
        help='a file with all references, line-by-line. '
             'if each document has more than one reference, '
             'divide them by \"|||\"')

    # Style Transfer
    parser.add_argument('--input_sent',
                        help='a file with all input sentences, line-by-line')

    return parser.parse_args()


def evaluate_style_transfer(args):
    scorer = StyleTransferScorer(align=args.align)

    scores = []
    for input_sent, hypo in zip(
            open(args.input_sent).readlines(),
            open(args.hypo).readlines()):
        input_sent, hypo = input_sent.strip(), hypo.strip()

        if input_sent == '' and hypo == '':
            continue
        scores.append(scorer.score(
            input_sent=input_sent,
            hypo=hypo,
            aspect=args.aspect,
            remove_stopwords=args.remove_stopwords))

    return scores


def evaluate_summarization(args):
    scorer = SummarizationScorer(align=args.align)

    scores = []
    for doc, refs, hypo in zip(
            open(args.doc).readlines(),
            open(args.refs).readlines(),
            open(args.hypo).readlines()):
        doc, refs, hypo = doc.strip(), refs.strip().split('|||'), hypo.strip()

        if doc == '' and hypo == '':
            continue
        scores.append(scorer.score(
            doc=doc,
            refs=refs,
            hypo=hypo,
            aspect=args.aspect,
            remove_stopwords=args.remove_stopwords))

    return scores


def evaluate_dialog(args):
    scorer = DialogScorer(align=args.align)

    scores = []
    for fact, dialog_history, hypo in zip(
            open(args.fact).readlines(),
            open(args.dialog_history).readlines(),
            open(args.hypo).readlines()):
        fact, dialog_history, hypo = \
            fact.strip(), dialog_history.strip(), hypo.strip()

        if fact == '' and dialog_history == '' and hypo == '':
            continue
        scores.append(scorer.score(
            fact=fact,
            dialog_history=dialog_history,
            hypo=hypo,
            aspect=args.aspect,
            remove_stopwords=args.remove_stopwords))

    return scores


def main():
    args = parse_args()

    if args.task == 'style_transfer':
        scores = evaluate_style_transfer(args)
    elif args.task == 'summarization':
        scores = evaluate_summarization(args)
    elif args.task == 'dialog':
        scores = evaluate_dialog(args)
    else:
        raise ValueError(f'\"task\" should be one of {TASKS}.')

    scores_file = open(args.scores_save_path, 'w')
    for score in scores:
        print(score, file=scores_file)

    print(f'#examples detected: {len(scores)}')
    print(f'{args.aspect} score: {sum(scores) / len(scores)}')
    print(f'example-wise scores are saved in {args.scores_save_path}')


if __name__ == '__main__':
    main()
