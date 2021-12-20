import spacy
from spacy.displacy import render
import pandas as pd
from collections import OrderedDict
from spacy.matcher import Matcher

en_model = spacy.load('en_core_web_md')
sentence = ("In 1541 Desoto wrote in his journal that the Pascagoula people"
            + "ranged as far north as the confluence of the Leaf and Chickasawhay river at 30.4, -88.5.")
parsed_sent = en_model(sentence)
print(parsed_sent.ents)

print(' '.join(['{}_{}'.format(tok, tok.tag_) for tok in parsed_sent]))

sentence = 'In 1541 Desoto wrote in his journal about the Pascagoula'
parsed_sent = en_model(sentence)
with open('pascagoula.html', 'w') as f:
    f.write(render(docs=parsed_sent, page=True, options=dict(compact=True)))


def token_dict(token):
    return OrderedDict(ORTH=token.orth_, LEMMA=token.lemma_, POS=token.pos_, TAG=token.tag_, DEP=token.dep_)


def doc_dataframe(doc):
    return pd.DataFrame([token_dict(tok) for tok in doc])


print(doc_dataframe(en_model("In 1541 Desoto met the Pascagoula.")))

pattern = [{'TAG': 'NNP', 'OP': '+'}, {'IS_ALPHA': True, 'OP': '*'}, {'LEMMA': 'meet'}, {'IS_ALPHA': True, 'OP': '*'},
           {'TAG': 'NNP', 'OP': '+'}]
doc = en_model("In 1541 Desoto met the Pascagoula")
matcher = Matcher(en_model.vocab)
matcher.add('met', pattern)
m = matcher(doc)
print(m)
doc = en_model("October 24: Lewis and Clark met their first Mandan Chief Big White.")
m = matcher(doc)[0]
print(m)
print(doc[m[1]:m[2]])
