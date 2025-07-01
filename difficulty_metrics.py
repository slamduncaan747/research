from lexicalrichness import LexicalRichness
import tiktoken

enc = tiktoken.get_encoding("gpt2")


def MTLD(example):
    text = example["text"]
    lex = LexicalRichness(text)
    return lex.mtld(threshold=0.72)


def TokenzeText(example):
    ids = enc.encode_ordinary(example["text"])
    return {"ids": ids, "len": len(ids)}
