"""Generate competition predictions and explanations for Attachments 3 and 4.

This entry point first checks that the local BERT checkpoint reproduces the
text features in Attachment 2. It never changes the supplied PKL files.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*arguments):
    script = Path(__file__).resolve().parent / arguments[0]
    subprocess.run([sys.executable, str(script), *map(str, arguments[1:])], check=True)


def check_csv(path, expected):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise ValueError(f'{path}: expected {expected} rows, found {len(rows)}')
    if len({row['sample_id'] for row in rows}) != expected:
        raise ValueError(f'{path}: sample_id values are not unique')
    for row in rows:
        value = float(row['intensity_pred'])
        if not -3 <= value <= 3:
            raise ValueError(f'{path}: intensity outside [-3,3]: {value}')
        if row['polarity_pred'] not in {'Negative', 'Neutral', 'Positive'}:
            raise ValueError(f'{path}: unknown polarity {row["polarity_pred"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='trained linear_baseline.npz')
    parser.add_argument('--bert_model', required=True, help='local bert-base-uncased')
    parser.add_argument('--reference_pkl', required=True,
                        help='Attachment 2 unaligned_50.pkl')
    parser.add_argument('--attachment3_dir', required=True)
    parser.add_argument('--attachment4_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='msa_submission_') as temporary:
        for number, source, expected in (
            (3, args.attachment3_dir, 30),
            (4, args.attachment4_dir, 20),
        ):
            prepared = Path(temporary) / f'attachment{number}'
            run('materialize_text.py', '--model', args.bert_model,
                '--reference_pkl', args.reference_pkl,
                '--input_dir', source, '--output_dir', prepared,
                '--device', args.device)
            explanations = output / f'attachment{number}_explanations.csv'
            predictions = output / f'attachment{number}_predictions.csv'
            run('linear_baseline.py', 'predict', '--model', args.model,
                '--input_dir', prepared, '--expected_count', expected,
                '--output_csv', explanations, '--submission_csv', predictions,
                '--vocab', Path(args.bert_model) / 'vocab.txt',
                '--video_dir', Path(source) / 'videos')
            check_csv(predictions, expected)
            check_csv(explanations, expected)
    print(f'Validated 30 Attachment 3 and 20 Attachment 4 results in {output}')


if __name__ == '__main__':
    main()
