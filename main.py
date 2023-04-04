import argparse
from src.wer_pos_weighted import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation de la sortie ASR pour la tâche de traduction en pictogrammes.")
    parser.add_argument("--out_asr", type=str, help="Fichier de sortie ASR produit par speechbrain")
    args = parser.parse_args()

    print("Commencement de l'évaluation ...")
    wer_by_tags_by_weights(args.out_asr)
    print("Evaluation terminée ! Le fichier de sortie output.csv a été généré.")
