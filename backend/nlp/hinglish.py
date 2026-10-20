"""Hinglish (code-mixed Hindi written in Roman script) support.

1. A lexicon of common Roman-Hindi words -> Devanagari (+ English gloss).
2. A rule-based Roman -> Devanagari transliterator for words not in the lexicon.
3. English words (decided with word-frequency data) are left in English, which is how
   students actually write mixed notes: "पौधे sunlight की मदद से food बनाते हैं".
"""
import re

from .text import LATIN_RE, tokenize

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    def zipf_frequency(word, lang):  # type: ignore
        return 0.0

# roman : devanagari : english gloss
_LEXICON_RAW = """
hai:है:is|hain:हैं:are|tha:था:was|thi:थी:was|the:थे:were|ho:हो:be|hota:होता:happens|hoti:होती:happens|hote:होते:happen
ka:का:of|ki:की:of|ke:के:of|ko:को:to|me:में:in|mein:में:in|mai:में:in|se:से:from|par:पर:on|pe:पे:on|aur:और:and
ya:या:or|bhi:भी:also|toh:तो:then|hi:ही:only|ek:एक:one|do:दो:two|teen:तीन:three|char:चार:four|yeh:यह:this|ye:ये:these
woh:वह:that|wo:वो:that|vo:वो:that|is:इस:this|us:उस:that|in:इन:these|un:उन:those|jo:जो:which|jis:जिस:which
jisme:जिसमें:in which|jisse:जिससे:by which|kya:क्या:what|kyu:क्यों:why|kyun:क्यों:why|kyunki:क्योंकि:because
lekin:लेकिन:but|magar:मगर:but|liye:लिए:for|lie:लिए:for|tak:तक:till|saath:साथ:with|sath:साथ:with|dwara:द्वारा:by
kar:कर:do|karta:करता:does|karti:करती:does|karte:करते:do|karna:करना:to do|kiya:किया:did|kiye:किए:did|krta:करता:does
gaya:गया:went|gayi:गई:went|gaye:गए:went|jata:जाता:is|jati:जाती:is|jate:जाते:are|jaata:जाता:is|jaati:जाती:is
raha:रहा:-ing|rahi:रही:-ing|rahe:रहे:-ing|apna:अपना:own|apni:अपनी:own|apne:अपने:own|kuch:कुछ:some|sab:सब:all
sabhi:सभी:all|nahi:नहीं:not|nahin:नहीं:not|nhi:नहीं:not|na:ना:not|bahut:बहुत:very|zyada:ज़्यादा:more|jyada:ज़्यादा:more
kam:कम:less|agar:अगर:if|jab:जब:when|tab:तब:then|wala:वाला:one|wale:वाले:ones|wali:वाली:one|hone:होने:being
banata:बनाता:makes|banati:बनाती:makes|banate:बनाते:make|banta:बनता:is made|banti:बनती:is made|bante:बनते:are made
bana:बना:made|banana:बनाना:to make|deta:देता:gives|deti:देती:gives|dete:देते:give|leta:लेता:takes|leti:लेती:takes
lete:लेते:take|milta:मिलता:gets|milti:मिलती:gets|milte:मिलते:get|badalta:बदलता:changes|badalti:बदलती:changes
badalte:बदलते:change|chahiye:चाहिए:should|sakta:सकता:can|sakti:सकती:can|sakte:सकते:can|mana:माना:assumed
kehte:कहते:call|kahte:कहते:call|kaha:कहा:said|matlab:मतलब:meaning|yani:यानी:that is|jaise:जैसे:like|aise:ऐसे:like this
waise:वैसे:like that|isliye:इसलिए:therefore|kaise:कैसे:how|kab:कब:when|kahan:कहाँ:where|kaun:कौन:who|koi:कोई:any
sabse:सबसे:most|pehle:पहले:first|baad:बाद:after|phir:फिर:then|fir:फिर:then|andar:अंदर:inside|bahar:बाहर:outside
upar:ऊपर:above|niche:नीचे:below|beech:बीच:between|madad:मदद:help|zaroori:ज़रूरी:important|jaruri:ज़रूरी:important
mahatvapurn:महत्वपूर्ण:important|prakriya:प्रक्रिया:process|urja:ऊर्जा:energy|paudhe:पौधे:plants|paudha:पौधा:plant
pani:पानी:water|paani:पानी:water|hawa:हवा:air|roshni:रोशनी:light|prakash:प्रकाश:light|surya:सूर्य:sun|suraj:सूरज:sun
khana:खाना:food|bhojan:भोजन:food|patti:पत्ती:leaf|pattiyan:पत्तियाँ:leaves|pattiyon:पत्तियों:leaves|mitti:मिट्टी:soil
jad:जड़:root|jadein:जड़ें:roots|ped:पेड़:tree|jeev:जीव:organism|jivan:जीवन:life|jeevan:जीवन:life|shareer:शरीर:body
sharir:शरीर:body|koshika:कोशिका:cell|rasayanik:रासायनिक:chemical|bhautik:भौतिक:physical|ganit:गणित:mathematics
vigyan:विज्ञान:science|itihas:इतिहास:history|desh:देश:country|sarkar:सरकार:government|log:लोग:people|samay:समय:time
din:दिन:day|raat:रात:night|saal:साल:year|naya:नया:new|nayi:नई:new|naye:नए:new|purana:पुराना:old|bada:बड़ा:big
badi:बड़ी:big|bade:बड़े:big|chhota:छोटा:small|chhoti:छोटी:small|chhote:छोटे:small|accha:अच्छा:good|acha:अच्छा:good
sahi:सही:correct|galat:गलत:wrong|samajh:समझ:understanding|samjho:समझो:understand|yaad:याद:remember|padh:पढ़:read
padhai:पढ़ाई:study|notes:नोट्स:notes|uska:उसका:its|uski:उसकी:its|uske:उसके:its|iska:इसका:its|iski:इसकी:its
iske:इसके:its|inka:इनका:their|unka:उनका:their|unki:उनकी:their|unke:उनके:their|hum:हम:we|tum:तुम:you|aap:आप:you
main:मैं:I|mujhe:मुझे:to me|hume:हमें:to us|humein:हमें:to us|dono:दोनों:both|bina:बिना:without|waala:वाला:one
"""

LEXICON: dict[str, tuple[str, str]] = {}
for _entry in _LEXICON_RAW.replace("\n", "|").split("|"):
    if _entry.strip():
        _roman, _dev, _gloss = _entry.strip().split(":")
        LEXICON[_roman] = (_dev, _gloss)

# Words that are ordinary English too; they are not counted as evidence of Hinglish.
_AMBIGUOUS = {"the", "is", "in", "us", "me", "main", "hi", "do", "par", "log", "na", "tab", "kam", "ho", "ya",
              "ye", "un", "pe", "char", "sab", "bad", "notes", "jo", "kar", "fir", "din", "mana", "bana", "tum"}
HINGLISH_MARKERS = set(LEXICON) - _AMBIGUOUS

# ---------- rule-based transliteration ----------
_CONSONANTS = [
    ("chh", "छ"), ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"),
    ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("gy", "ज्ञ"), ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "त"),
    ("d", "द"), ("n", "न"), ("p", "प"), ("f", "फ़"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"),
    ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("q", "क़"), ("x", "क्स"),
]
# vowel: (independent form, matra form)
_VOWELS = [
    ("aa", "आ", "ा"), ("ee", "ई", "ी"), ("ii", "ई", "ी"), ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"), ("ai", "ऐ", "ै"),
    ("au", "औ", "ौ"), ("ei", "ए", "े"), ("a", "अ", ""), ("e", "ए", "े"), ("i", "इ", "ि"), ("o", "ओ", "ो"), ("u", "उ", "ु"),
]
HALANT = "्"
ANUSVARA = "ं"


def _match(word: str, i: int, table):
    for item in table:
        if word.startswith(item[0], i):
            return item
    return None


def transliterate_word(word: str) -> str:
    """Best-effort Roman -> Devanagari for a single Hindi word (e.g. 'paudhe' -> 'पौधे')."""
    w, out, i, pending, last_vowel = word.lower(), [], 0, False, ""
    while i < len(w):
        # 'n' after a vowel becomes the nasal anusvara before a consonant (sundar -> सुंदर)
        # or at the end of a word after a long vowel (hain -> हैं, nahin -> नहीं).
        if w[i] == "n" and not pending and out:
            at_end = i + 1 == len(w)
            before_consonant = not at_end and _match(w, i + 1, _VOWELS) is None and w[i + 1] != "h"
            if before_consonant or (at_end and last_vowel not in ("a", "i", "u")):
                out.append(ANUSVARA)
                i += 1
                continue
        consonant = _match(w, i, _CONSONANTS)
        if consonant:
            if pending:
                out.append(HALANT)
            out.append(consonant[1])
            pending = True
            i += len(consonant[0])
            continue
        vowel = _match(w, i, _VOWELS)
        if vowel:
            roman, independent, matra = vowel
            end = i + len(roman) == len(w)
            if pending:
                if roman == "a" and end:
                    matra = "ा"  # Hinglish writers spell final 'आ' as 'a': karta, pata, khana
                elif roman == "i" and end:
                    matra = "ी"  # final 'i' is usually long: pani, nadi
                out.append(matra)
            else:
                out.append(independent)
            pending = False
            last_vowel = roman
            i += len(roman)
            continue
        out.append(w[i])
        pending = False
        i += 1
    return "".join(out)


_ENGLISH_SHAPE = re.compile(r"c(?!h)|x|q|tion$|sion$|ology$|ism$|ide$|ase$|ose$|oid$|ium$|ph[^aeiou]|[^aeiou]y[^aeiou]|ll$|ss$")


def best_transliteration(word: str) -> str:
    """Romanised Hindi is ambiguous ('khana' could be खना or खाना). Try spelling variants of the short
    vowels and keep the one that is most frequent in real Hindi text (wordfreq)."""
    w = word.lower()
    slots = [i for i, ch in enumerate(w) if ch in "aiu" and 0 < i and w[i - 1] not in "aeiou"
             and (i + 1 >= len(w) or w[i + 1] not in "aeiou")][:5]
    best, best_score = transliterate_word(w), -1.0
    for mask in range(1 << len(slots)):
        chars = list(w)
        for bit, i in enumerate(slots):
            if mask >> bit & 1:
                chars[i] = {"a": "aa", "i": "ee", "u": "oo"}[w[i]]
        candidate = transliterate_word("".join(chars))
        score = zipf_frequency(candidate, "hi")
        if score > best_score + 1e-9:
            best, best_score = candidate, score
    return best


def is_english_word(word: str) -> bool:
    """Decide whether a Roman-script word in Hinglish text is English (keep) or Hindi (transliterate)."""
    lw = word.lower()
    if lw in LEXICON:
        return False
    if len(lw) <= 2 or not lw.isalpha():
        return True
    en = zipf_frequency(lw, "en")
    hi = zipf_frequency(best_transliteration(lw), "hi")
    if hi >= 2.5 and hi >= en:
        return False
    if en >= 2.5 or _ENGLISH_SHAPE.search(lw):
        return True
    return False  # unknown to both lists: most likely a Hindi word spelt in Roman script


def hinglish_score(text: str) -> tuple[int, int]:
    """(number of Hinglish marker words, number of Latin words)."""
    latin = [t.lower() for t in tokenize(text) if LATIN_RE.search(t)]
    return sum(1 for t in latin if t in HINGLISH_MARKERS), len(latin)


# Ambiguous words that are usually Hindi inside a Hinglish sentence ("Photosynthesis me ...")
_WEAK_ENGLISH = {"me", "hi", "par", "na", "kam", "ya", "ye", "un", "pe", "sab", "jo", "kar", "fir", "mana", "bana",
                 "tum", "ho"}


def _classify(word: str) -> str:
    lw = word.lower()
    if lw in HINGLISH_MARKERS:
        return "hi"
    if lw in _AMBIGUOUS:
        return "amb"
    return "en" if is_english_word(word) else "hi"


def _convert_sentence(sentence: str, changed: list[str]) -> str:
    matches = list(re.finditer(r"[A-Za-z]+", sentence))
    classes = [_classify(m.group(0)) for m in matches]
    has_marker = any(c == "hi" for c in classes)
    for i, c in enumerate(classes):
        if c != "amb":
            continue
        lw = matches[i].group(0).lower()
        neighbours = [classes[j] for j in (i - 1, i + 1) if 0 <= j < len(classes)]
        if lw in _WEAK_ENGLISH:
            classes[i] = "hi" if has_marker else "en"
        else:  # strong English words ('the', 'is', 'in' ...) need Hindi on both sides
            classes[i] = "hi" if neighbours and all(n == "hi" for n in neighbours) else "en"

    out, last = [], 0
    for m, c in zip(matches, classes):
        out.append(sentence[last:m.start()])
        word = m.group(0)
        if c == "hi":
            lw = word.lower()
            dev = LEXICON[lw][0] if lw in LEXICON else best_transliteration(word)
            changed.append(dev)
            out.append(dev)
        else:
            out.append(word)
        last = m.end()
    out.append(sentence[last:])
    return "".join(out)


def to_devanagari(text: str) -> tuple[str, list[str]]:
    """Convert the Hindi words of a Hinglish text to Devanagari, keep English words. Returns (text, changed_words)."""
    changed: list[str] = []
    # split but keep the separators so the original layout survives
    pieces = re.split(r"([.!?।\n]+)", text)
    converted = [p if re.fullmatch(r"[.!?।\n]+", p or "x") else _convert_sentence(p, changed) for p in pieces]
    return "".join(converted), changed


def word_gloss_english(text: str) -> str:
    """Very rough offline fallback: replace known Hindi words with English glosses."""
    def repl(match: re.Match) -> str:
        word = match.group(0)
        lw = word.lower()
        if lw in LEXICON and not is_english_word(word):
            return LEXICON[lw][1]
        return word

    return re.sub(r"[A-Za-z]+", repl, text)
