from flair.data import Sentence
from flair.models import SequenceTagger
import csv
import spacy

tagger = SequenceTagger.load("flair/upos-multi-fast")
level_1 = ["NOUN", "VERB", "AUX", "ADJ", "PROPN"]
level_2 = ["NOUN", "VERB", "AUX", "ADJ", "PROPN", "PRON"]
model = spacy.load("fr_dep_news_trf")


def clean(line, info_asr=False):
    line = line.split(";")
    if info_asr:
        line = " ".join(line)
    else:
        line = " ".join(line).lower()
    line = line.split()
    line = "\t".join(line)
    return line


def process_data(namef):  # useful for CER, EmbER and SemDist
    id_file = []
    ref_asr = []
    hyp_asr = []
    info_asr_errors = []
    wers = []
    with open(namef, "r", encoding="utf8") as basefile:
        lines = basefile.readlines()
        for i in range(0, len(lines), 5):
            if lines[i].split(" ")[1] == "%WER":
                idfile = lines[i].split(" ")[0].split(",")[0]
                wer = float(lines[i].split(" [")[0].split("%WER ")[1])
            if wer != 0.00:
                wers.append(wer)
                id_file.append(idfile)
                i += 1
                line = clean(lines[i])
                ref_asr.append(line)
                i += 1
                line = clean(lines[i], True)
                info_asr_errors.append(line)
                i += 1
                line = clean(lines[i])
                hyp_asr.append(line)
    return id_file, ref_asr, hyp_asr, info_asr_errors, wers


def mapping():
    mapper = {"<eps>": "<eps>", '': ''}
    with open("mapping.txt", "r", encoding="utf8") as file:
        for ligne in file:
            ligne = ligne[:-1].split("\t")
            mapper[ligne[0]] = ligne[1]
    return mapper


def convert(pos, mapper):
    pos_new = []
    for i in range(len(pos)):
        pos_new.append(mapper[pos[i]])
    print("\t".join(pos_new))


def POS(sentence):
    """
    Input :
        sentence = "BONJOUR JE VAIS BIEN <eps> <eps>"
    Output :
        pos = #part-of-speech of sentence with <eps> symbol instead of pos
    """
    # The next part is useful to delete <eps> from the sentence but to keep it in memory
    sentence = sentence.split("\t")
    eps_index = get_list_index("<eps>", sentence)
    eps_index.sort(reverse=True)
    for ind in eps_index:
        del sentence[ind]
    s = []
    for x in sentence:
        if str(x) == 'j':
            s.append('je')
        elif str(x) == 'd':
            s.append('de')
        elif str(x) == "l":
            s.append('le')
        else:
            s.append(str(x))
    sentence = " ".join(s)

    # Prediction of POS
    sentence = Sentence(sentence)
    tagger.predict(sentence)
    pos = getPosTxt(sentence)

    # Adding <eps> in POS sentence
    pos = pos.split(" ")
    eps_index.sort()
    for ind in eps_index:
        pos.insert(ind, "<eps>")
    pos = "\t".join(str(x) for x in pos)
    return pos


def get_list_index(elem, l):
    return [i for i, d in enumerate(l) if d == elem]


def getPosTxt(sentence):
    """
    Input :
        sentence = #Format Sentence with associated POS
    Output :
        pos = "PRON VERB ADJ"
    """
    pos = ""
    for i in range(len(sentence.labels)):
        pos += sentence.labels[i].value + " "
    pos = pos[:-1]
    return pos


def get_level1_num_words(pos_tag_ref):
    return sum(1 for i in pos_tag_ref if i in level_1)


def get_level2_num_words(pos_tag_ref):
    return sum(1 for i in pos_tag_ref if i in level_2)


def get_index(liste, el):
    return [index for index, value in enumerate(liste) if value == el]


def get_tags_from_index(pos_tags_ref, index):
    return [pos_tags_ref[i] for i in index]


def get_index_for_substitution(asr_tags, ref_asr, hyp_asr):
    all_index = get_index(asr_tags, "S")
    final_index = []
    for el in all_index:
        lemmas = model(ref_asr[el] + ' ' + hyp_asr[el])
        f_lemmas = [token.lemma_ for token in lemmas]
        same = all(x == f_lemmas[0] for x in f_lemmas)
        if not same:
            final_index.append(el)
    return final_index


def get_score_by_level(tags_with_errors, num_words_1, num_words_2, num_words_3):
    errors_level1 = 0
    errors_level2 = 0
    errors_level3 = 0
    for el in tags_with_errors:
        if el in level_1:
            errors_level1 += 1
        elif el in level_2:
            errors_level2 += 1
        else:
            errors_level3 += 1

    if errors_level1 == 0:
        wer_level1_score = 0
    else:
        wer_level1_score = (1.5 * errors_level1) / num_words_1 * 100
    if errors_level2 == 0 and errors_level1 == 0:
        wer_level2_score = 0
    else:
        wer_level2_score = ((1.5 * errors_level1) + (1.3 * errors_level2)) / num_words_2 * 100
    wer_level3_score = ((1.5 * errors_level1) + (1.3 * errors_level2) + errors_level3) / num_words_3 * 100
    return [str(wer_level1_score), str(wer_level2_score), str(wer_level3_score)]


def pos_tag_from_asr(out_asr):
    id_file, ref_asr, hyp_asr, info_asr_errors, wers = process_data(out_asr)
    pos_tag_ref = []
    pos_tag_hyp = []
    for i in range(len(ref_asr)):
        pos_tag_ref.append(POS(ref_asr[i]))
        pos_tag_hyp.append(POS(hyp_asr[i]))
    return id_file, ref_asr, hyp_asr, info_asr_errors, wers, pos_tag_ref, pos_tag_hyp


def calculate_wer_by_pos_and_level(id_file, ref_asr, hyp_asr, info_asr_errors, pos_tag_ref):
    final_wer_by_tags = {}

    for i in range(len(ref_asr)):
        asr_tags = info_asr_errors[i].split('\t')
        pos_tags_ref_split = pos_tag_ref[i].split('\t')
        ref_asr_split = ref_asr[i].split('\t')
        hyp_asr_split = hyp_asr[i].split('\t')
        subtitutions_index = get_index_for_substitution(asr_tags, ref_asr_split, hyp_asr_split)
        insertion_index = get_index(asr_tags, "I")
        deletion_index = get_index(asr_tags, "D")
        all_errors = subtitutions_index + insertion_index + deletion_index
        tags_with_errors = get_tags_from_index(pos_tags_ref_split, all_errors)
        final_wer_by_tags[id_file[i]] = get_score_by_level(tags_with_errors, get_level1_num_words(pos_tags_ref_split),
                                                           get_level2_num_words(pos_tags_ref_split),
                                                           len(pos_tags_ref_split))

    return final_wer_by_tags


def save_in_csv(id_file, ref_asr, hyp_asr, info_asr_errors, wers, pos_tag_ref, pos_tag_hyp, final_wer_by_tags):
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(ref_asr)):
            wers_by_tags_by_level = "\t".join(final_wer_by_tags[id_file[i]])
            writer.writerow([id_file[i]+"\t"+str(wers[i])+"\t"+wers_by_tags_by_level])
            writer.writerow([ref_asr[i]])
            writer.writerow([pos_tag_ref[i]])
            writer.writerow([info_asr_errors[i]])
            writer.writerow([pos_tag_hyp[i]])
            writer.writerow([hyp_asr[i]])
            writer.writerow([])


def wer_by_tags_by_weights(out_asr):
    id_file, ref_asr, hyp_asr, info_asr_errors, wers, pos_tag_ref, pos_tag_hyp = pos_tag_from_asr(out_asr)
    final_wer_by_tags = calculate_wer_by_pos_and_level(id_file, ref_asr, hyp_asr, info_asr_errors, pos_tag_ref)
    save_in_csv(id_file, ref_asr, hyp_asr, info_asr_errors, wers, pos_tag_ref, pos_tag_hyp, final_wer_by_tags)


if __name__ == '__main__':
    wer_by_tags_by_weights("../data/Exemple/test_cecile.txt")
