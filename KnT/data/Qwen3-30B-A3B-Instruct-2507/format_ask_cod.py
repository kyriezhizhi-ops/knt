import os
from langcodes import Language
import argparse
from data_format import KEYWORDS, TRIGGER_SENTS

SUPPORT_LANGS = TRIGGER_SENTS.keys()
FIXED_AUXILIARY_LANGS = ["fr", "de", "pt"]

def create_multilingual_demo_dict():
    demo_dict = {}
    all_langs = list(SUPPORT_LANGS)
    
    for src_lng in all_langs:
        for tgt_lng in all_langs:
            if src_lng == tgt_lng:
                continue
            
            aux_langs = [l for l in FIXED_AUXILIARY_LANGS if l not in {src_lng, tgt_lng}]
            demo_dict[(src_lng, tgt_lng)] = []
            for sent_idx in range(len(TRIGGER_SENTS[src_lng])):
                tri_sent = TRIGGER_SENTS[src_lng][sent_idx]
                keywords = []
                lang_keywords = {lang: KEYWORDS[lang][sent_idx] for lang in all_langs}
                for kw_idx in range(len(lang_keywords[src_lng])):
                    kw_parts = []
                    kw_parts.append(f'"{lang_keywords[src_lng][kw_idx]}"')
                    kw_parts.append(f'"{lang_keywords[tgt_lng][kw_idx]}"')
                    for aux_lang in FIXED_AUXILIARY_LANGS:
                        if aux_lang not in {src_lng, tgt_lng} and kw_idx < len(lang_keywords[aux_lang]):
                            kw_parts.append(f'"{lang_keywords[aux_lang][kw_idx]}"')
                    keywords.append(" means ".join(kw_parts))
                keywords_str = ", ".join(keywords)
                demo_dict[(src_lng, tgt_lng)].append((tri_sent, keywords_str))
    return demo_dict

demo_dict = create_multilingual_demo_dict()
def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', "--workspace", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), help="Workspace dir")
    parser.add_argument('-tn', "--test-name", type=str, required=True, help="wmt22/wmt21/...")
    parser.add_argument('-s', "--src", type=str, required=True, help='source lang')
    parser.add_argument('-t', "--tgt", type=str, required=True, help='target lang')
    parser.add_argument('-m', "--modelname", type=str, required=True, help='model-name')
    return parser.parse_args()

def main(args):
    workspace = args.workspace
    data_dir = os.path.join(workspace, "data")
    raw_dir = os.path.join(data_dir, "raw")
    format_dir = os.path.join(data_dir, "format")
    test_name = args.test_name
    model_name = args.modelname
    src = args.src
    tgt = args.tgt
    auxiliary_langs = [l for l in FIXED_AUXILIARY_LANGS if l not in {src, tgt}]
    for lang in [src, tgt] + auxiliary_langs:
        if lang not in SUPPORT_LANGS:
            raise ValueError(
                f"Unsupported language: {lang}. Supported languages are: {SUPPORT_LANGS}"
            )
    auxiliary_langs = [lang for lang in auxiliary_langs if lang not in {src, tgt}]
    src_full = Language.make(language=src).display_name()
    tgt_full = Language.make(language=tgt).display_name()
    aux_full_names = [Language.make(language=lang).display_name() for lang in auxiliary_langs]
    all_langs = [src, tgt] + auxiliary_langs
    all_langs_full = [src_full, tgt_full] + aux_full_names
    try:
        with open(os.path.join(raw_dir, f"{test_name}.{src}-{tgt}.{src}")) as test_src_f:
            test_src_lines = [l.strip() for l in test_src_f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Source file not found: {test_name}.{src}-{tgt}.{src} in {raw_dir}"
        )
    folder_name = f"{test_name}.{src}-{tgt}"
    folder_path = os.path.join(format_dir, model_name, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    out_file_path = os.path.join(folder_path, f"{test_name}.{src}-{tgt}.{src}.cod")
    demos = demo_dict[(src, tgt)]
    with open(out_file_path, 'w',encoding='utf-8') as out_f:
        for id, src_line in enumerate(test_src_lines):
            num_demos = min(3, len(demos))  
            selected_demos = demos[:num_demos]
            all_items = selected_demos + [(src_line, None)]
            prompt_lst = []
            for it in all_items:
                it_src, it_kw = it
                if len(all_langs_full) > 1:
                    lang_list_str = ", ".join(all_langs_full[:-1]) + " and " + all_langs_full[-1]
                else:
                    lang_list_str = all_langs_full[0]
                instruction = (
                    f"Extract the keywords from the given {src_full} sentence and provide their translations into {tgt_full}, French, German and Portuguese.Only output the result, using 'means' to separate translations across languages and enclosing each word in double quotes.\n"
                )
                s = (
                    instruction
                    + f"{src_full}: {it_src}\n\n"
                    + (
                        f"Chain-of-Dictionary:{it_kw}"
                        if it_kw else "Chain-of-Dictionary:"
                    )
                )
                prompt_lst.append(s)
            prompt = "\n\n".join(prompt_lst) 
            out_f.write(
                f"{id:04}\n"
                f"{prompt}\n\n\n"
            )

if __name__ == "__main__":
    args = parse_args()
    main(args)