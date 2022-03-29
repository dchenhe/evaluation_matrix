import argparse
import os
import time
import json
import numpy as np
from utils import *




class Scorer:
    """ Support ROUGE-1,2,L, BERTScore, MoverScore, PRISM, BARTScore """

    def __init__(self, ref, hypo,src, device='cuda:0', output = ".", multi_ref=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including src, ref_summ, ref_summs, and sys_summ.
        """
        self.SRC = read_file_to_list(src)
        self.multi_ref = multi_ref
        self.device = device
        self.REF = read_file_to_list(ref)
        self.HYPO = read_file_to_list(hypo)
        self.output = output + "/output.txt"

    def score(self, metrics):
        """ metrics: list of metrics """
        for metric_name in metrics:
            if metric_name == 'bert_score':
                from bert_score import score


                print(f'BERTScore setup finished. Begin calculating BERTScore.')

                
                ref_lines = [line for line in self.REF]

                sys_lines = [line for line in self.HYPO]

                cur = {}

                P, R, F1 = score(sys_lines, ref_lines, lang="en", verbose=True)
                cur["F1:"] = np.sum(F1.mean().numpy())
                print("Bert_Score:")
                print(cur)

                with open(self.output,"a") as fd:
                    print("Bert_Score:",file=fd)
                    print(cur,file=fd)


                print(f'Finished calculating BERTScore')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                """ Vanilla BARTScore, BARTScore-CNN, BARTScore-CNN-Para """
                from bart_score import BARTScorer

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='./bart_large_cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='./bart_large_cnn')
                    bart_scorer.load()
                else:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='./bart_large')
                print(f'BARTScore setup finished. Begin calculating BARTScore.')

                start = time.time()
                # Keep capitalization, detokenize everything
                cur = {}
                ref_lines = [detokenize(line) for line in self.REF]
                sys_lines = [detokenize(line) for line in self.HYPO]

                src_lines = [detokenize(line.split("</s>")[1]) for line in self.SRC]

                faith = np.array(bart_scorer.score(src_lines, ref_lines, batch_size=4))
                #ref_hypo = np.array(bart_scorer.score(ref_lines, sys_lines, batch_size=4))
                fs = np.array(bart_scorer.score(sys_lines, ref_lines, batch_size=4))
                rs = np.array(bart_scorer.score(ref_lines, sys_lines, batch_size=4))
                f1 = (fs+rs)/2
                cur["F1"] = np.log(np.sum(np.exp(f1))/f1.size)
                cur["Faithfulness"] = np.log(np.sum(np.exp(faith))/faith.size)
                #np.log(np.sum(np.exp(avg_f))/avg_f.size)
                # harm_f = (ref_hypo * hypo_ref) / (ref_hypo + hypo_ref)
                # ref_hypo = np.sum(np.exp(ref_hypo))/ref_hypo.size
                # path = "./bart_score.txt"
                # os.makedirs(os.path.dirname(path), exist_ok=True)
                # with open(path,"w") as fd:
                #     # fd.write("ref_hypo_bart_score"+ref_hypo+"\n")
                #     fd.write("hypo_ref_bart_score: {}".format(ref_hypo))
                #     # fd.write("avg_f_bart_score"+avg_f+"\n")
                #     # fd.write("harm_f_bart_score"+harm_f+"\n")
                print("Bart_Score:")
                print(cur)
                with open(self.output,"a") as fd:
                    print("Bart_Score:",file=fd)
                    print(cur,file=fd)

                print(f'Finished calculating BARTScore')

            else:
                raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--ref', type=str, required=True,
                        help='ref summary')
    parser.add_argument('--hypo', type=str, required=True,
                        help='hypo summary')
    parser.add_argument('--src', type=str, required=True,
                        help='src')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--multi_ref', action='store_true', default=False,
                        help='Whether we are using multiple references to calculate scores.')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--mover_score', action='store_true', default=False,
                        help='Whether to calculate MoverScore')
    parser.add_argument('--rouge', action='store_true', default=False,
                        help='Whether to calculate ROUGE')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--output',  default=".",
                        help='the path to output')
    args = parser.parse_args()
    print("Begin to set up!")
    scorer = Scorer(args.ref, args.hypo,args.src, args.device, args.output,args.multi_ref)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.mover_score:
        METRICS.append('mover_score')
    if args.rouge:
        METRICS.append('rouge')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prism:
        METRICS.append('prism')

    scorer.score(METRICS)


if __name__ == '__main__':
    main()