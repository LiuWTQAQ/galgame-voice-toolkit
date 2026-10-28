"""日语敏感词加权表 + 娇喘/非语言发声启发式。

模块只负责「文本/声学特征 → 分数增量 + 中文理由」，不做阈值判定（阈值在 filter.py）。

设计原则
--------
1. 露骨解剖词、露骨行为词、体液/妊娠词权重高，**单个命中就足以到剔除线**；
2. 「気持ちいい」「だめ」「やめて」这类日常也常见的词权重低（0.7），
   单独出现只到 0.7（远低于 review 线 1.5），只有多个不同词组合累计才会被剔除；
3. 娇喘/非语言发声用启发式单独加分：纯假名短句、小写假名（ぁぃぅぇぉっ）、
   同字连续重复、低 avg_logprob、高 no_speech_prob、转写为空的长句、VAD 吞句；
4. 全部在归一化文本上匹配（片假名→平假名、去标点/空白/长音符），
   因此「オチンチン」「おちんちんー」都能命中同一词条。

对外接口
--------
- ``normalize(text) -> str``: 归一化文本
- ``match_terms(text) -> list[TermHit]``: 词表命中
- ``classify(text) -> TextFeatures``: 文本层面的全部特征（命中/娇喘/幻觉）
- ``score_text(text) -> tuple[float, list[str]]``: 文本层面的分数与理由
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
_KATAKANA_START = 0x30A1
_KATAKANA_END = 0x30F6
_KATA_TO_HIRA = str.maketrans(
    {chr(c): chr(c - 0x60) for c in range(_KATAKANA_START, _KATAKANA_END + 1)}
)

#: 匹配前剔除的字符：空白/标点/长音符/波浪线/引号。剔除长音符让「はぁー」命中「はぁ」。
_STRIP_RE = re.compile(
    r"[\s\u3000、。，．,.!?！？…‥・「」『』（）()【】\[\]〜~ー－\-—―:;：；\"'”’*※]"
)

#: 汉字（CJK 统一表意文字）。出现汉字基本可以认为是有语义的台词。
_KANJI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
#: 假名（含小写假名、片假名）
_KANA_RE = re.compile(r"[\u3041-\u309f\u30a1-\u30fa]")


def normalize(text: str) -> str:
    """归一化：NFKC → 片假名转平假名 → 去标点/空白/长音符 → 小写。

    归一化后同一个词的不同写法（片假名/长音/空格）会落到同一个字符串，
    词表因此只需要写一份平假名形态。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_KATA_TO_HIRA)
    return _STRIP_RE.sub("", text).lower()


# --------------------------------------------------------------------------- #
# 词表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Term:
    """一条敏感词规则。"""

    surface: str  # 归一化后的词面
    weight: float  # 命中一次的分值
    category: str  # 分类 key
    label: str  # 中文分类名（进 reasons）


#: 分类 → 中文名。顺序即 reasons 里的展示优先级。
CATEGORY_LABELS: dict[str, str] = {
    "organ": "露骨解剖",
    "act": "露骨行为",
    "fluid": "体液妊娠",
    "body": "性化身体部位",
    "context": "性语境",
    "soft": "暧昧词",
}

#: 分类基础权重。高权重分类单个命中即可到 drop 线（4.0）。
CATEGORY_WEIGHTS: dict[str, float] = {
    "organ": 6.0,
    "act": 5.0,
    "fluid": 4.0,
    "body": 2.2,
    "context": 1.5,
    "soft": 0.7,
}

#: 词表：分类 → 词面（平假名/汉字/拉丁均可，代码里统一 normalize）。
_TERMS: dict[str, tuple[str, ...]] = {
    # ---- 露骨解剖（6.0，单个即剔除）------------------------------------- #
    "organ": (
        "おちんちん",
        "ちんちん",
        "ちんこ",
        "ちんぽ",
        "ぺにす",
        "まんこ",
        "おまんこ",
        "まんまん",
        "膣内",
        "膣",
        "子宮",
        "卵巣",
        "乳首",
        "乳輪",
        "亀頭",
        "肉棒",
        "肉芽",
        "くりとりす",
        "陰茎",
        "陰核",
        "陰部",
        "陰唇",
        "精巣",
        "前立腺",
        "gすぽっと",
        "あなる",
    ),
    # ---- 露骨行为（5.0，单个即剔除）------------------------------------- #
    "act": (
        "せっくす",
        "せっくすする",
        "中出し",
        "膣内射精",
        "挿入",
        "挿れる",
        "挿れて",
        "ふぇら",
        "ふぇらちお",
        "ぱいずり",
        "手こき",
        "おなにー",
        "射精",
        "顔射",
        "精飲",
        "口内射精",
        "れいぷ",
        "輪姦",
        "りょうじょく",
        "調教",
        "痴漢",
        "痴女",
        "淫乱",
        "淫ら",
        "はつじょう",
        "交尾",
        "種付け",
        "孕ませ",
        "妊娠させ",
        "堕胎",
        "中絶",
        "強姦",
        "近親相姦",
        "獣姦",
        "寝取られ",
        "乱交",
        "はめる",
        "はめ倒す",
        "犯す",
        "犯して",
        "突かれる",
        "騎乗位",
        "正常位",
        "後背位",
        "くんに",
        "しっくすないん",
        "あなるせっくす",
        "そーぷ",
        "売春",
        "援交",
        "でりへる",
        "しゃぶる",
        "しゃぶって",
        "ぺろぺろ",
        "れろれろ",
    ),
    # ---- 体液 / 妊娠（4.0，单个即剔除）---------------------------------- #
    "fluid": (
        "精液",
        "精子",
        "ざーめん",
        "あいえき",
        "らぶじゅーる",
        "潮吹き",
        "潮を吹く",
        "母乳",
        "搾乳",
        "受精",
        "孕む",
        "妊娠した",
        "妊娠して",
    ),
    # ---- 性化身体部位 / 情色评价（2.2，两个才到 drop 线）---------------- #
    "body": (
        "おっぱい",
        "おしり",
        "お尻",
        "尻",
        "胸",
        "乳房",
        "巨乳",
        "爆乳",
        "美乳",
        "太もも",
        "ふともも",
        "あそこ",
        "いやらしい",
        "やらしい",
        "えっち",
        "すけべ",
        "変態",
        "えろ",
        "むらむら",
        "欲情",
        "性的",
        "全裸",
        "半裸",
        "下着",
        "ぱんつ",
        "ぶらじゃー",
        "かんび",
        "きょにゅう",
    ),
    # ---- 性语境（1.2，需要组合）---------------------------------------- #
    "context": (
        "気持ちいい",
        "気持ちよく",
        "気持ちよすぎ",
        "きもちいい",
        "きもちよく",
        "いっちゃう",
        "いかせて",
        "いかされて",
        "感じちゃ",
        "感じて",
        "感じる",
        "濡れて",
        "濡れる",
        "濡らし",
        "熱い",
        "震える",
        "ふるえる",
        "我慢できない",
        "我慢できな",
        "止まらない",
        "とまらない",
        "壊れる",
        "こわれる",
        "締まる",
        "締め付け",
        "吸い付く",
        "舐める",
        "舐めて",
        "舐め",
        "きす",
        "接吻",
        "抱いて",
        "抱かれ",
        "触って",
        "触られ",
        "挿れ",
        "出して",
        "いっぱい",
        "もっと",
        "敏感",
        "びくびく",
        "ぞくぞく",
        "どきどき",
        "恥ずかしい",
        "はしたない",
        "破廉恥",
        "だらしない",
        "蕩け",
        "とろけ",
        "とろとろ",
        "奥",
        "深い",
        "激しい",
        "はげしい",
    ),
    # ---- 暧昧词（0.7，单独出现基本不判定）------------------------------- #
    "soft": (
        "だめ",
        "やめて",
        "やめっ",
        "いや",
        "やだ",
        "いく",
        "いくよ",
        "はぁ",
        "ふぅ",
        "ん",
        "あ",
        "あぁ",
        "はぁっ",
        "そこ",
        "ここ",
        "ちょっと",
    ),
}

#: 最少参与子串匹配的归一化长度。单字（あ/ん）会命中所有句子，直接丢弃；
#: 两字词（だめ/やめて/はぁ）保留，靠排除短语和低权重控制误杀。
_MIN_SUBSTR_LEN = 2

#: 预编译：分类 → tuple[Term]
_TERM_DB: dict[str, tuple[Term, ...]] = {}
for _cat, _surfaces in _TERMS.items():
    _items: list[Term] = []
    for _s in _surfaces:
        _n = normalize(_s)
        if len(_n) >= _MIN_SUBSTR_LEN:
            _items.append(Term(_n, CATEGORY_WEIGHTS[_cat], _cat, CATEGORY_LABELS[_cat]))
    _TERM_DB[_cat] = tuple(_items)

#: 强分类：词表命中直接累加（单个词就够高）。
_STRONG_CATEGORIES = ("organ", "act", "fluid", "body")

#: 需要「组合加分」的低权重分类：同一句出现多个不同词才累计奖励。
_COMBO_CATEGORIES = ("context", "soft")
_COMBO_STEP = 0.6
_COMBO_CAP = 1.8

#: 明确的排除短语：命中则抵消对应词的权重（避免「ハメを外す」这类误杀）。
_EXCLUSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("はめる", ("はめをはず", "はめはず")),
    ("いく", ("いくら", "いくつ", "いくつか")),
    ("奥", ("奥さん", "おくさん", "奥様")),
    ("深い", ("深いこと", "深い話")),
)


# --------------------------------------------------------------------------- #
# 词表命中
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TermHit:
    """一次词表命中。"""

    surface: str
    weight: float
    category: str
    label: str


def match_terms(text: str) -> list[TermHit]:
    """返回文本命中的全部词条（同一词条只算一次，已排除排除短语）。"""
    norm = normalize(text)
    if not norm:
        return []
    hits: list[TermHit] = []
    seen: set[str] = set()
    for cat in ("organ", "act", "fluid", "body", "context", "soft"):
        for term in _TERM_DB.get(cat, ()):  # type: ignore[arg-type]
            if term.surface in seen:
                continue
            pos = norm.find(term.surface)
            if pos < 0:
                continue
            skip = False
            for ex_surface, ex_phrases in _EXCLUSIONS:
                if term.surface == normalize(ex_surface):
                    window = norm[max(0, pos - 2) : pos + len(term.surface) + 4]
                    if any(p in window for p in (normalize(x) for x in ex_phrases)):
                        skip = True
                        break
            if skip:
                continue
            seen.add(term.surface)
            hits.append(term)
    # 去重：短词是长词子串时（いやらしい 里的 やらしい）不算两次，避免同一处文本被重复计分。
    return [h for h in hits if not any(h.surface != o.surface and h.surface in o.surface for o in hits)]


# --------------------------------------------------------------------------- #
# 娇喘 / 非语言发声
# --------------------------------------------------------------------------- #
#: 娇喘常见字符集：元音、拨音、促音、小写假名、鼻音、气声。
_MOAN_CHARS = set("あぁぃぅぇぉおうっんー゛はふへひゃゅょわをやよ")
#: 小写假名（含促音）。大量出现基本只剩娇喘/语气词。
_SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョ")
#: 「喘」的形态标记：小写假名 / 促音 / 拨音 / 长音。没有这些标记的纯假名句
#: （おはよう / ありがとう）是正常台词，不能按娇喘计分。
_MOAN_MARKS = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョー゛")
#: 少量常见短语气词/寒暄：即使字符全落在娇喘字符集里也判为正常台词。
_COMMON_SHORT_WORDS: frozenset[str] = frozenset(
    {
        "おはよう",
        "おやすみ",
        "ありがとう",
        "ごめん",
        "ごめんなさい",
        "こんにちは",
        "こんばんは",
        "おかえり",
        "いってきます",
        "ただいま",
        "いただきます",
        "ごちそうさま",
        "おめでとう",
        "がんばって",
        "だいじょうぶ",
        "すごい",
        "かわいい",
        "うれしい",
        "たのしい",
        "さようなら",
        "うん",
        "ううん",
        "はい",
        "いいえ",
        "あっ",
        "ああ",
        "えっ",
        "ええ",
        "おっ",
        "おお",
        "そう",
        "えと",
        "あの",
        "その",
    }
)
#: 同字连续重复（あああ / んんん / ぅぅぅ）。
_REPEAT_RUN_RE = re.compile(r"(.)\1{2,}")
#: 笑声：うふふ / あはは / ふふっ / はっはっ 等。这些是正常语气，不能被当成娇喘。
#: 注意也要覆盖「はっはっはっ」「へへっ」「ほほっ」这类**带促音**的重复笑，
#: 它们不会被「同字连续」规则命中，但字符全在喘声集合里，容易误判成娇喘。
_LAUGH_RE = re.compile(
    r"(うふふ|ふふふ|あはは|ははは|うはは|えへへ|ふふっ|あはっ|うふー|ふはは"
    r"|へへ|ほほ|はっはっ|へっへっ|ふはっ|んふ|うひひ|にひひ|ふふん)"
)
#: 组合短语：模棱两可词只有跟特定搭配同时出现时才升级为露骨语境。
_PHRASE_RULES: tuple[tuple[str, float, str], ...] = (
    (r"(だめ|やめて|いや|こら)[^ぁ-ん]{0,4}(いっちゃう|いってしまう|いくの|いかされる)", 2.0, "拒绝+要去了"),
    (r"(いっちゃう|いってしまう|いきそう)[^ぁ-ん]{0,4}(だめ|やめて)", 2.0, "要去了+拒绝"),
    (r"(気持ちいい|きもちいい)[^ぁ-ん]{0,6}(だめ|やめて|いっちゃう|いく)", 1.5, "快感+拒绝/要去了"),
    (r"(吸って|舐めて|しゃぶって)[^ぁ-ん]{0,4}(出して|いって|だめ)", 1.5, "口交语境"),
    (r"(おっぱい|おしり|胸)[^ぁ-ん]{0,4}(大き|柔らか|気持ち|舐め|触)", 1.2, "身体部位+性化修饰"),
    (r"(中|奥)[^ぁ-ん]{0,3}(出す|出して|当たる)", 1.2, "体内语境"),
)


@dataclass
class TextFeatures:
    """文本层面的全部特征，供 filter 汇总打分。"""

    hits: list[TermHit] = field(default_factory=list)
    moan_score: float = 0.0
    moan_reasons: list[str] = field(default_factory=list)
    hallucination_reasons: list[str] = field(default_factory=list)
    kana_only: bool = False
    has_kanji: bool = False
    moan_only: bool = False
    small_kana: int = 0
    max_repeat: int = 0


def _max_repeat_len(text: str) -> int:
    """最长同字连续重复长度（「うぅぅぅ」→ 4）。"""
    m = _REPEAT_RUN_RE.search(text)
    return (len(m.group(0)) if m else 1)


def _moan_only(norm: str) -> bool:
    """整句是否为纯娇喘/语气词：短、无汉字、含喘声标记、且不是常见寒暄。"""
    if not norm or len(norm) > 10:
        return False
    if _KANJI_RE.search(norm) or norm in _COMMON_SHORT_WORDS:
        return False
    if _LAUGH_RE.search(norm):
        return False
    return any(ch in _MOAN_MARKS for ch in norm) and all(ch in _MOAN_CHARS for ch in norm)


def _repeat_phrase_len(norm: str) -> int:
    """最长的「1~2 字片段重复 ≥3 次」片段长度（0 = 没有）。

    只认「含喘声标记的片段」（あっ / はぁ / ん）或单字重复（あああ），
    避免把「そうそうそう」这类正常口语重复当成呻吟。
    """
    best = 0
    for size in (1, 2):
        i = 0
        while i + size * 3 <= len(norm):
            unit = norm[i : i + size]
            if size == 1 or any(ch in _MOAN_MARKS for ch in unit):
                j = i + size
                while norm[j : j + size] == unit:
                    j += size
                count = (j - i) // size
                if count >= 3:
                    best = max(best, count * size)
                    i = j
                    continue
            i += 1
    return best


def moan_features(text: str) -> tuple[float, list[str]]:
    """娇喘/非语言发声启发式打分（只看文本形态，不看声学概率）。

    关键点：**必须同时满足「短」「无汉字」「含喘声标记」**才可能是娇喘。
    早期版本用「假名占比 ≥0.75」判断，结果所有正常日语台词都被判成娇喘
    （日语正常句子本来就是假名占多数），已改为「喘声标记占比」并加寒暄白名单。

    Args:
        text: ASR 文本（原始或已归一化都可以）。

    Returns:
        ``(score, reasons)``；score 只是文本侧的加分，声学特征由 filter 叠加。
    """
    norm = normalize(text)
    if not norm:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []

    has_kanji = bool(_KANJI_RE.search(norm))
    small = sum(1 for ch in norm if ch in _SMALL_KANA)
    marks = sum(1 for ch in norm if ch in _MOAN_MARKS)
    max_repeat = _max_repeat_len(norm)
    repeat_phrase = _repeat_phrase_len(norm)
    short = len(norm) <= 10
    # 整句（归一化后）只由喘声字符组成，且至少有一个喘声标记，且不是常见寒暄
    moan_only = _moan_only(norm)

    if moan_only:
        score += 2.0 + min(1.0, 0.25 * marks)
        reasons.append("整句为娇喘/语气词")
        if marks >= 2:
            # 「んっ、あっ」这类两个以上气声标记的纯呻吟，比单声「はぁ」更露骨
            score += 0.4
            reasons.append("多气声标记")
    if max_repeat >= 3:
        score += 1.0
        reasons.append(f"同字连续×{max_repeat}")
    if repeat_phrase >= 4 and repeat_phrase > max_repeat:
        # 「あっ! あっ! あっ!」这种非连续重复，是成人台词的典型形态；越长越确定
        bonus = min(2.5, 0.4 + 0.15 * repeat_phrase)
        score += bonus
        reasons.append(f"重复喘息片段×{repeat_phrase}")
    if short and not has_kanji and (small >= 2 or max_repeat >= 3):
        score += 0.8
        reasons.append("纯假名短句")
    if short and not has_kanji and marks >= 2 and marks * 2 >= len(norm):
        score += 0.6
        reasons.append("气声假名占比高")
    if "ー" in text and len(norm) <= 8:
        score += 0.3
        reasons.append("拖长音")
    if _LAUGH_RE.search(norm):
        # 笑声几乎总是正常台词，直接压到 0.5 以内，避免「あははは」被当成娇喘。
        return min(0.5, score * 0.2), reasons + ["笑声(非娇喘)降权"]
    return score, reasons


#: Whisper 在静音/气声上常见的幻觉句（faster-whisper 小模型 japanese）。
_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録",
    "高評価",
    "字幕",
    "ご覧いただき",
    "おやすみなさい",
    "おめでとうございます",
)


def hallucination_reasons(text: str) -> list[str]:
    """识别 Whisper 幻觉/重复退化文本（这类文本不代表真实台词，只提示不强判）。"""
    norm = normalize(text)
    if not norm:
        return []
    out: list[str] = []
    for pat in _HALLUCINATION_PATTERNS:
        if normalize(pat) in norm:
            out.append("疑似幻觉句")
            break
    # 同一 4~6 字片段重复 3 次以上：典型的幻觉/结巴退化
    if len(norm) >= 12:
        for size in (4, 5, 6):
            if len(norm) < size * 3:
                continue
            gram = norm[:size]
            if norm.count(gram) >= 3:
                out.append("重复退化文本")
                break
    return out


def classify(text: str) -> TextFeatures:
    """提取文本层面的全部特征。"""
    norm = normalize(text)
    feats = TextFeatures()
    feats.hits = match_terms(text)
    feats.moan_score, feats.moan_reasons = moan_features(text)
    feats.hallucination_reasons = hallucination_reasons(text)
    feats.kana_only = bool(_KANA_RE.search(norm)) and not _KANJI_RE.search(norm)
    feats.has_kanji = bool(_KANJI_RE.search(norm))
    feats.moan_only = _moan_only(norm)
    feats.small_kana = sum(1 for ch in norm if ch in _SMALL_KANA)
    feats.max_repeat = _max_repeat_len(norm) if norm else 1
    return feats


# --------------------------------------------------------------------------- #
# 文本打分
# --------------------------------------------------------------------------- #
def _combo_bonus(hits: list[TermHit]) -> tuple[float, list[str]]:
    """低权重词组合加分：同一句出现 2 个以上不同暧昧/语境词才奖励。"""
    n = sum(1 for h in hits if h.category in _COMBO_CATEGORIES)
    if n < 2:
        return 0.0, []
    bonus = min(_COMBO_CAP, (n - 1) * _COMBO_STEP)
    return bonus, [f"暧昧词组合×{n}"]


def phrase_score(text: str) -> tuple[float, list[str]]:
    """组合短语加分：低权重词只有跟特定搭配同时出现才升级。"""
    norm = normalize(text)
    if not norm:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []
    for pattern, weight, label in _PHRASE_RULES:
        if re.search(pattern, norm):
            score += weight
            reasons.append(f"{label}+{weight:.1f}")
    return score, reasons


def score_text(text: str) -> tuple[float, list[str]]:
    """文本层面打分：词表加权 + 娇喘启发式 + 组合加分。

    Returns:
        ``(score, reasons)``，reasons 为中文短语，可直接进 CSV 的 reasons 列。
    """
    feats = classify(text)
    if not normalize(text):
        return 0.0, []

    score = 0.0
    reasons: list[str] = []

    # 强分类（解剖/行为/体液/身体部位）累加；弱分类（语境/暧昧词）只取最高分，
    # 避免「ちょっと」「そこ」这类日常词把普通台词刷成 review。
    by_cat: dict[str, float] = {}
    for hit in feats.hits:
        prev = by_cat.get(hit.category, 0.0)
        if hit.category in _STRONG_CATEGORIES:
            by_cat[hit.category] = prev + hit.weight
        else:
            by_cat[hit.category] = max(prev, hit.weight)
        reasons.append(f"{hit.label}「{hit.surface[:8]}」+{hit.weight:.1f}")
    score += sum(by_cat.values())

    bonus, bonus_reasons = _combo_bonus(feats.hits)
    if bonus:
        score += bonus
        reasons.extend(f"{r}+{bonus:.1f}" for r in bonus_reasons)

    p_score, p_reasons = phrase_score(text)
    if p_score:
        score += p_score
        reasons.extend(p_reasons)

    # 快感词 + 喘息气声同框：成人台词最典型的组合（「あっ…気持ちいい…」）
    norm = normalize(text)
    marks = sum(1 for ch in norm if ch in _MOAN_MARKS)
    if marks >= 2 and any(("気持" in h.surface or "きもち" in h.surface) for h in feats.hits):
        score += 1.0
        reasons.append("快感词+喘息声+1.0")

    if feats.moan_score:
        score += feats.moan_score
        reasons.append("娇喘特征(" + "/".join(feats.moan_reasons) + f")+{feats.moan_score:.1f}")

    for r in feats.hallucination_reasons:
        score += 0.3
        reasons.append(f"{r}+0.3")

    return round(score, 2), reasons


#: 高权重分类：校准报告里统计「显式词命中率」用。
EXPLICIT_CATEGORIES: tuple[str, ...] = ("organ", "act", "fluid")


def has_explicit(text: str) -> bool:
    """文本是否命中露骨解剖/行为/体液词（用于统计命中率，不参与阈值判定）。"""
    return any(h.category in EXPLICIT_CATEGORIES for h in match_terms(text))


def term_summary(text: str) -> str:
    """把命中词压成一行（调试/报告用，最多 3 个）。"""
    return ", ".join(f"{h.label}:{h.surface}" for h in match_terms(text)[:3])
