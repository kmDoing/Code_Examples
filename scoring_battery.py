"""
This script is designed to score medical documents.
It contains 5 scoring metrics:
ROUGE-all, BERTscore, BLUErt,
and QuickUMLS (code only, needs MetaMat file to run)
https://pypi.org/project/quickumls/
This file does not run. It is only as an example. If I was going to score files, I would find
some new implementations of these or other metrics.
"""
from pyrouge import Rouge155
from bert_score import BERTScorer
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from quickumls import QuickUMLS

def main():
    run_rouge()
    run_bert()
    run_bleurt()
    run_quickUMLS()


def run_rouge(truth, pred):
    # Run the rouge comparison between the truth and prediction
    # truth: some_name.001.txt, pred: some_name.A.001.txt
    r = Rouge155
    r.system_dir = pred
    r.model_dir = truth
    r.system_filename_pattern = r'clef_task.(\d+).txt'
    r.model_filename_pattern = r'clef_task.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    print(output)


def run_bert():
    # Run the BERTscore for pred given truth
    # Example texts
    print("Running BERTscore")
    reference = "This is a reference text example."
    candidate = "This is a candidate text example."
    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


def run_bleurt():
    # Run BLEUrt score
    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

    references = ["a bird chirps by the window", "this is a random sentence"]
    candidates = ["a bird chirps by the window", "this looks like a random sentence"]

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
        res = model(**inputs).logits.flatten().tolist()
    print(res)
    # [0.9604414105415344, 0.8080050349235535]


def run_quickUMLS():
    # Run medical fact extraction comparison
    quickumls_fp = 'path to UMLS files'
    # There are other attributes that can be set, but we are using the defaults
    accepted_semtypes = 'set this to limit semantic types'
    matcher = QuickUMLS(quickumls_fp, accepted_semtypes)
    text = "The ulna has dislocated posteriorly from the trochlea of the humerus."
    matcher.match(text, best_match=True, ignore_syntax=False)

if __name__ == "__main__":
    main()