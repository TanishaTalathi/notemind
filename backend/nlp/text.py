"""Shared text utilities: sentence splitting, tokenising and stopwords for English, Hindi and Hinglish."""
import re

# Devanagari block minus the danda (।) and double danda (॥), which are sentence punctuation.
DEVANAGARI = "ऀ-ॣ०-ॿ"
WORD_RE = re.compile(rf"[A-Za-z0-9{DEVANAGARI}]+(?:['’][A-Za-z]+)?")
DEVANAGARI_RE = re.compile(rf"[{DEVANAGARI}]")
LATIN_RE = re.compile(r"[A-Za-z]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")

EN_STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can cannot could did do does doing down during each either etc few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just let me more most my
myself no nor not now of off on once only or other ought our ours ourselves out over own per same she should so
some such than that the their theirs them themselves then there these they this those through thus to too under
until up upon very was we were what when where which while who whom why will with within without would you your
yours yourself yourselves via e.g i.e may might must shall one two three many much every another
use uses used using make makes made called known include includes including important also like well get gets
""".split())

HI_STOPWORDS = set("""
का की के को में मे से पर है हैं था थी थे हो होता होती होते होना और या भी तो ही एक यह ये वह वे इस उस इन उन
जो जिस जिसे जिसमें जिनमें कि लिए लिये तक साथ द्वारा कर करता करती करते करना किया किये गया गई गए जाता जाती
जाते जा रहा रही रहे अपना अपनी अपने कुछ सभी सब नहीं न ना बहुत अधिक कम तथा एवं अथवा यदि क्योंकि इसलिए जब तब
वाले वाला वाली होने रूप प्रकार बाद पहले बीच ऊपर नीचे अंदर बाहर
""".split())

# Roman-script Hindi function words commonly found in Hinglish notes.
HINGLISH_STOPWORDS = set("""
hai hain tha thi the ho hota hoti hote ka ki ke ko me mein mai se par aur ya bhi toh to hi ek yeh ye vo woh wo
is us in un jo jis kya kyu kyun kyunki lekin magar ki liye lie tak saath dwara kar karta karti karte karna kiya
gaya gayi jata jati jate raha rahi rahe apna apni apne kuch sab nahi nhi na bahut zyada kam agar jab tab wala wale
wali hone hota
""".split())

STOPWORDS = EN_STOPWORDS | HI_STOPWORDS | HINGLISH_STOPWORDS


def split_sentences(text: str) -> list[str]:
    """Split notes into sentences. Handles '.', '?', '!', the Hindi danda '।' and line breaks."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    parts = [p.strip(" -•*\t") for p in SENT_SPLIT_RE.split(text)]
    return [p for p in parts if len(tokenize(p)) >= 2]


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def tokens_lower(text: str) -> list[str]:
    return [t.lower() for t in tokenize(text)]


def is_devanagari(word: str) -> bool:
    return bool(DEVANAGARI_RE.search(word))


def script_ratio(text: str) -> float:
    """Share of letters that are Devanagari (0 = all Latin, 1 = all Devanagari)."""
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    return dev / (dev + lat) if dev + lat else 0.0


def is_stopword(word: str) -> bool:
    return word.lower() in STOPWORDS


def word_count(text: str) -> int:
    return len(tokenize(text))
