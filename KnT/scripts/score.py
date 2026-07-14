import os
import argparse
from comet import load_from_checkpoint, download_model
import os
import json
import threading
from bleurt import score as bleurt_score
import logging
from sacrebleu.metrics import BLEU, CHRF

os.environ["TRANSFORMERS_OFFLINE"] = "1"


comet_model_mapping = {
    "wmt21-comet-qe-da": "wmt21-comet-qe-da/checkpoints/model.ckpt",
    "Unbabel/wmt22-comet-da": "wmt22-comet-da/checkpoints/model.ckpt",
    "Unbabel/XCOMET-XL": "XCOMET-XL/checkpoints/model.ckpt", # 新增这一行
    "BLEURT-20": "bleurt_model/BLEURT-20"
}

def wait_until_path_exist(path):
    while not os.path.isdir(path):
        pass
    return

def comet(**kwargs):
    sys_lines = kwargs["sys_lines"]
    src_lines = kwargs["src_lines"]
    ref_lines = kwargs["ref_lines"]
    comet_model_name = kwargs["comet_model_name"]
    comet_saving_dir = kwargs["comet_saving_dir"]
    comet_cache_dir = kwargs["comet_cache_dir"]
    batch_size = kwargs["batch_size"]
    cpu = kwargs["cpu"]
    cache_file = os.path.join(comet_cache_dir, 'comet_cache.json')
    
    wait_until_path_exist(comet_saving_dir)

    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    data = []
    new_sys_lines, new_src_lines, new_ref_lines = [], [], []
    for sys, src, ref in zip(sys_lines, src_lines, ref_lines):
        cache_key = json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_src_lines.append(src)
            new_ref_lines.append(ref)
            data.append({"mt": sys, "src": src, "ref": ref})

    logging.info(f"COMET cache info: {len(sys_lines)-len(data)}/{len(sys_lines)}")
    if data:
        if comet_model_name in comet_model_mapping:
            model_path = "..../comet_model/wmt22-comet-da/checkpoints/model.ckpt"
            comet_model = load_from_checkpoint(model_path,local_files_only=True,reload_hparams=True)
        else:
            # model_path = download_model(comet_model_name, saving_directory=comet_saving_dir)
            model_path = "..../comet_model/wmt22-comet-da/checkpoints/model.ckpt"
            comet_model = load_from_checkpoint(model_path,local_files_only=True, reload_hparams=True)
        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=0 if cpu else 1)
        scores = model_output.scores

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            for (sys, src, ref), score in zip(zip(new_sys_lines, new_src_lines, new_ref_lines), scores):
                cache_key = json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)
                cache[cache_key] = score

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)

        final_scores = [cache[json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)] for sys, src, ref in zip(sys_lines, src_lines, ref_lines)]

    return sum(final_scores)/len(final_scores)


def xcomet(**kwargs):
    sys_lines = kwargs["sys_lines"]
    src_lines = kwargs["src_lines"]
    ref_lines = kwargs["ref_lines"]

    xcomet_model_name = "Unbabel/XCOMET-XL" 
    
    comet_saving_dir = kwargs["comet_saving_dir"]
    comet_cache_dir = kwargs["comet_cache_dir"]
    batch_size = kwargs["batch_size"]
    cpu = kwargs["cpu"]
    cache_file = os.path.join(comet_cache_dir, 'xcomet_cache.json')
    
    wait_until_path_exist(comet_saving_dir)
    cache_lock = threading.Lock()

    with cache_lock:
        cache = json.load(open(cache_file, 'r')) if os.path.exists(cache_file) else {}

    data = []
    new_sys_lines, new_src_lines, new_ref_lines = [], [], []
    for sys, src, ref in zip(sys_lines, src_lines, ref_lines):
        cache_key = json.dumps((xcomet_model_name, sys, src, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_src_lines.append(src)
            new_ref_lines.append(ref)
            data.append({"mt": sys, "src": src, "ref": ref})

    logging.info(f"XCOMET cache info: {len(sys_lines)-len(data)}/{len(sys_lines)}")
    
    if data:

            # 兜底路径（你之前提供的路径）
        model_path = "..../comet_model/XCOMET-XL/checkpoints/model.ckpt"

        comet_model = load_from_checkpoint(model_path, local_files_only=True, reload_hparams=True)
        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=0 if cpu else 1)
        scores = model_output.scores

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f: cache = json.load(f)
            else: cache = {}

            for (sys, src, ref), score in zip(zip(new_sys_lines, new_src_lines, new_ref_lines), scores):
                cache_key = json.dumps((xcomet_model_name, sys, src, ref), ensure_ascii=False)
                cache[cache_key] = score
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        cache = json.load(open(cache_file, 'r'))
        final_scores = [cache[json.dumps((xcomet_model_name, sys, src, ref), ensure_ascii=False)] for sys, src, ref in zip(sys_lines, src_lines, ref_lines)]

    return sum(final_scores)/len(final_scores)


def comet_qe(**kwargs):
    sys_lines = kwargs["sys_lines"]
    src_lines = kwargs["src_lines"]
    comet_qe_model_name = kwargs["comet_qe_model_name"]
    comet_saving_dir = kwargs["comet_saving_dir"]
    comet_cache_dir = kwargs["comet_cache_dir"]
    batch_size = kwargs["batch_size"]
    cache_file = os.path.join(comet_cache_dir, 'comet_qe_cache.json')

    wait_until_path_exist(comet_saving_dir)

    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    data = []
    new_sys_lines, new_src_lines = [], []
    for sys, src in zip(sys_lines, src_lines):
        cache_key = json.dumps((comet_qe_model_name, sys, src), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_src_lines.append(src)
            data.append({"mt": sys, "src": src, "ref": None})

    logging.info(f"COMET-QE cache info: {len(sys_lines)-len(data)}/{len(sys_lines)}")
    if data:
        if comet_model_name in comet_model_mapping:
            comet_model = load_from_checkpoint(os.path.join(comet_saving_dir, comet_model_mapping[comet_model_name]),local_files_only=True, reload_hparams=True)
        else:
            # model_path = download_model(comet_model_name, saving_directory=comet_saving_dir)
            model_path = "/..../comet_model/wmt21-comet-qe-da/checkpoints/model.ckpt"
            comet_model = load_from_checkpoint(model_path, local_files_only=True, reload_hparams=True)
        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=1)
        scores = model_output.scores

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            for (sys, src), score in zip(zip(new_sys_lines, new_src_lines), scores):
                cache_key = json.dumps((comet_qe_model_name, sys, src), ensure_ascii=False)
                cache[cache_key] = score

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)

        final_scores = [cache[json.dumps((comet_qe_model_name, sys, src), ensure_ascii=False)] for sys, src in zip(sys_lines, src_lines)]

    return sum(final_scores)/len(final_scores)

def bleurt(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    bleurt_cache_dir = kwargs["bleurt_cache_dir"]
    bleurt_ckpt = kwargs["bleurt_ckpt"]
    batch_size = kwargs["batch_size"]
    comet_saving_dir = kwargs["comet_saving_dir"] # 确保传入了这个参数
    cache_file = os.path.join(bleurt_cache_dir, 'bleurt_cache.json')
    cache_lock = threading.Lock()

    # --- 修改点 1: 确保目录存在，防止 open() 报错 ---
    if not os.path.exists(bleurt_cache_dir):
        os.makedirs(bleurt_cache_dir, exist_ok=True)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    new_sys_lines, new_ref_lines = [], []
    for sys, ref in zip(sys_lines, ref_lines):
        cache_key = json.dumps((sys, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_ref_lines.append(ref)

    logging.info(f"BLEURT cache info: {len(sys_lines)-len(new_sys_lines)}/{len(sys_lines)}")
    assert len(new_sys_lines) == len(new_ref_lines)
    if len(new_sys_lines) > 0:
        bleurt_model = bleurt_score.LengthBatchingBleurtScorer(bleurt_ckpt)
        scores = bleurt_model.score(references=new_ref_lines, candidates=new_sys_lines, batch_size=batch_size)

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            for (sys, ref), score in zip(zip(new_sys_lines, new_ref_lines), scores):
                cache_key = json.dumps((sys, ref), ensure_ascii=False)
                cache[cache_key] = score

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        final_scores = [cache[json.dumps((sys, ref), ensure_ascii=False)] for sys, ref in zip(sys_lines, ref_lines)]

    return sum(final_scores)/len(final_scores)

def bleu(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    tgt_lang = kwargs["tgt_lang"]
    cache_dir = kwargs.get("traditional_cache_dir", None)
    if cache_dir is None:
        # 回退到默认目录，避免传参遗漏
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'traditional')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'bleu_cache.json')

    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    # 准备计算的数据
    new_sys, new_ref = [], []
    for sys, ref in zip(sys_lines, ref_lines):
        # 缓存键包含语言，确保不同语言设置不会混用
        cache_key = json.dumps(("bleu", sys, ref, tgt_lang), ensure_ascii=False)
        if cache_key not in cache:
            new_sys.append(sys)
            new_ref.append(ref)

    logging.info(f"BLEU cache info: {len(sys_lines)-len(new_sys)}/{len(sys_lines)}")
    if new_sys:
        if tgt_lang == "zh":
            scorer = BLEU(tokenize="zh")
        else:
            scorer = BLEU()
        # sacrebleu 的 corpus_score 接受列表，这里逐句计算保留句子级分数
        scores = [scorer.corpus_score([s], [[r]]).score for s, r in zip(new_sys, new_ref)]

        with cache_lock:
            # 重新读取可能被其他线程更新的缓存
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            for (s, r), score in zip(zip(new_sys, new_ref), scores):
                cache_key = json.dumps(("bleu", s, r, tgt_lang), ensure_ascii=False)
                cache[cache_key] = score
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    # 获取所有句子的最终分数
    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        final_scores = [cache[json.dumps(("bleu", s, r, tgt_lang), ensure_ascii=False)]
                        for s, r in zip(sys_lines, ref_lines)]
    return final_scores


def chrf(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    cache_dir = kwargs.get("traditional_cache_dir", None)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'traditional')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'chrf_cache.json')

    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    new_sys, new_ref = [], []
    for sys, ref in zip(sys_lines, ref_lines):
        cache_key = json.dumps(("chrf", sys, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys.append(sys)
            new_ref.append(ref)

    logging.info(f"chrF++ cache info: {len(sys_lines)-len(new_sys)}/{len(sys_lines)}")
    if new_sys:
        # 使用 chrF++ 默认设置：word_order=2, beta=2
        chrf_scorer = CHRF(word_order=2)
        scores = [chrf_scorer.corpus_score([s], [[r]]).score for s, r in zip(new_sys, new_ref)]

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            for (s, r), score in zip(zip(new_sys, new_ref), scores):
                cache_key = json.dumps(("chrf", s, r), ensure_ascii=False)
                cache[cache_key] = score
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        final_scores = [cache[json.dumps(("chrf", s, r), ensure_ascii=False)]
                        for s, r in zip(sys_lines, ref_lines)]
    return final_scores

def readlines(file_path):
    if not file_path:
        return []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

def check_equal_num_lines(paths):
    if len(paths) <= 1:
        return True
    else:
        return all([len(readlines(p)) == len(readlines(paths[0])) for p in paths[1:]])

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sys", nargs='+', required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--ref", type=str)
    parser.add_argument("--tgt-lang", type=str)
    parser.add_argument("--comet-qe-model-name", type=str, default="wmt21-comet-qe-da")
    parser.add_argument("--comet-model-name", type=str, default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-cache-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'comet'))
    parser.add_argument("--comet-saving-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ''))
    parser.add_argument("--bleurt-ckpt", type=str, default="..../bleurt_model/BLEURT-20")
    parser.add_argument("--bleurt-cache-dir", type=str, default="....../code/Knt_ours/cache/bleurt")
    parser.add_argument("--metric", type=str, choices=["comet", "xcomet", "comet_qe", "bleurt", "bleu", "chrf"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--traditional-cache-dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'traditional'),
                    help="Cache directory for traditional metrics (BLEU, chrF++)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys_file_paths = args.sys
    src_file_path = args.src
    ref_file_path = args.ref
    tgt_lang = args.tgt_lang
    comet_qe_model_name = args.comet_qe_model_name
    comet_model_name = args.comet_model_name
    comet_cache_dir = args.comet_cache_dir
    comet_saving_dir = args.comet_saving_dir
    bleurt_ckpt = args.bleurt_ckpt
    bleurt_cache_dir = args.bleurt_cache_dir
    metric = args.metric
    batch_size = args.batch_size
    cpu = args.cpu
    traditional_cache_dir = args.traditional_cache_dir

    assert tgt_lang or metric != "bleu", "BLEU need to specify target language. (--tgt-lang xx)"

    sys_lines_lst = [
        readlines(v)
        for v in sys_file_paths
    ]
    src_lines = readlines(src_file_path)
    ref_lines = readlines(ref_file_path)

    if metric != "comet_qe":
        assert all([len(src_lines) == len(ref_lines)] + [len(sys_lines) ==  len(src_lines) for sys_lines in sys_lines_lst])
    else:
        assert all([len(sys_lines) ==  len(src_lines) for sys_lines in sys_lines_lst])

    scorer = eval(metric)
    scores = [
        scorer(**{
            "sys_lines": sys_lines,
            "src_lines": src_lines,
            "ref_lines": ref_lines,
            "comet_qe_model_name": comet_qe_model_name,
            "comet_model_name": comet_model_name,
            "comet_cache_dir": comet_cache_dir,
            "comet_saving_dir": comet_saving_dir,
            "bleurt_ckpt": bleurt_ckpt,
            "bleurt_cache_dir": bleurt_cache_dir,
            "batch_size": batch_size,
            "cpu": cpu,
            "tgt_lang": tgt_lang,
            "traditional_cache_dir": args.traditional_cache_dir   # 新增
        })
        for sys_lines in sys_lines_lst
    ]
    scale_metrics = {"comet", "xcomet", "comet_qe", "bleurt"}

    for n, s in zip(sys_file_paths, scores):
        if metric in ("bleu", "chrf"):
            avg = sum(s) / len(s) if s else 0.0
            print(f"{os.path.basename(n)}: {avg:.2f}")
        else:
            print(f"{os.path.basename(n)}: {s*100:.2f}")