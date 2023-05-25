# importing required modules
import argparse
import nltk
import spacy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from typing import Tuple, List
import PyPDF2
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
# from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain, pretty_goal_tree
import re
import logging

LOCAL_TEXT_PATH = 'manual_text/'
LOCAL_PATH = 'manuals/'
IF_ = 'if'
THEN = 'then'
NEVER = 'never'
BC = 'BECAUSE'

AND = ' and '
THAT = ' that '
ANDS = [AND, THAT]

OR = ' or '
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']
SUBJECTS = ['NN', 'NNP', 'NNS']

TO_BE = ['is', 'am', 'are', 'was', 'were', 'be', 'being', 'been']
TO_HAVE = ['have', 'has', 'had', 'having']
NOTS = ['not', 'never']

RE_SPLITTERS = '[:,.]'
CONJS = [AND, OR]
MAX_WORDS = 25  # Sometimes sentences don't get split well...

KEY_PHRASES = ['blind spot', 'traffic light', 'traffic signal', 'safety belt', 'blind spot']

# For accuracy
# nlp = spacy.load('en_core_web_trf')

# For efficiency
nlp = spacy.load('en_core_web_sm')


def read_manual(state:str='MA', file_name='MA_Drivers_Manual.pdf', rule_file:str=""):
    """
    File located at 
    MA: https://driving-tests.org/wp-content/uploads/2020/03/MA_Drivers_Manual.pdf
    CA: https://www.dmv.ca.gov/portal/file/california-driver-handbook-pdf/
    """
    if state == 'CA':
        file_name = 'CA_driving_handbook.pdf'
    pdfFile = open(LOCAL_PATH + file_name, 'rb')
    # creating a pdf reader object 
    pdfReader = PyPDF2.PdfFileReader(pdfFile)

    # printing number of pages in pdf file 
    MAX_PAGES = pdfReader.numPages
    #    MAX_PAGES = 10
    START_PAGE = 84 # This starts from the rules of the road for MA.
    END_PAGE = MAX_PAGES-1 #START_PAGE+40 # MAX_PAGES
    all_rules = []
    all_sentences = []

    """
    For Mass 82-124
    """
    for page in range(START_PAGE, END_PAGE):
        pageObj = pdfReader.getPage(page)
        pageText = pageObj.extractText()
        
        # if page == START_PAGE:
        #     print(pageText)
        (rules, sentences) = extract_if_then(pageText)
        all_rules.extend(rules)
        all_sentences.extend(sentences)

    # closing the pdf file object
    print("Found %d potential rules" % len(all_rules))
    pdfFile.close()

    # if there is a rule file, then write it to file.
    if rule_file:
        write_to_text_file(all_sentences, rule_file)
    return all_rules

def write_to_text_file(sentences: List, rule_file: str):
    with open(rule_file, 'w') as f:
        for sentence in sentences:
            f.write(sentence)
    f.close()


def extract_if_then(page_text: str):
    """
    Check for rule keywords in text
    """
    rule_counter = 0
    rules = []
    all_sentences = [] # For printing to file
    counter = 0

    # sometimes in reading the pdf we will get non-ascii characters
    new_val = page_text.encode("ascii", "ignore")
    updated_text = new_val.decode()
    sentences = updated_text.split('.')

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        if IF_ in tokens and len(tokens) < MAX_WORDS:
            words = [word for word in tokens if word.isalpha()]
            stripped = words[0]
            for item in words[1::]:
                stripped+= " %s"%item
            # TODO: check sentence
            rule = extract_rule(sentence)
            # if("is" in sentence):
               
            if not 'None' in str(rule):  # and containsNumber(sentence):
                logging.debug("Root it %s" % sentence.strip())
                logging.debug("  Rule is:  %s" % rule)
                counter += 1
                print(sentence)
                print(rule)
                rules.append(rule)
                all_sentences.append(stripped+"\n")
    return (rules, all_sentences)


def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False


def extract_rule(sentence) -> str:
    """
    Tries to extract an IF/THEN rule from a sentence.  Returns it in the form: IF(if triples), THEN(then triples)
    """
    logging.debug("What is the sentence %s" % sentence)
    if_then = re.split(RE_SPLITTERS, sentence)
    # sometimes if is the last part:
    try:
        if_clause, then_clause = set_if_clause(if_then)

        if_triples = make_triples_from_phrase(if_clause)
        then_triples = make_triples_from_phrase(then_clause)
        return 'IF %s, THEN %s' % (if_triples, then_triples)
    except TypeError:
        print("error")


def set_if_clause(clauses) -> Tuple:
    """
    Sets the if clause and the then clause for a rule.
    - If there are two parts, then it will return the if then
    """
    logging.debug("I'm here with %s" % clauses)
    if len(clauses) == 2:
        if IF_ in clauses[0].lower():
            return tuple(clauses)
        else:
            return clauses[1], clauses[0]
    elif len(clauses) == 1:  # It didn't get separated
        logging.debug("Didn't split on regex, trying to split on if or then keyword")
        if IF_ in clauses[0]:
            all_tokens = clauses[0].split(IF_)
            then_clause = all_tokens[0]
            full_if = ""
            for part in all_tokens[1::]:
                full_if += part.strip() + ' '
            return full_if.strip(), then_clause.strip()
    else:  # put the commas back together
        full_then = ""
        for item in clauses[1::]:
            full_then += item
        return clauses[0], full_then


def make_triples_from_phrase(phrase: str, full_phrase: str = ""):
    """
    Struggled with this one. So I think we need to find all the occurences
    Keeping a full phrase in case....
    """
    logging.debug("  Making triples for %s" % phrase)
    if AND in phrase or OR in phrase or THAT in phrase:
        tokens = word_tokenize(phrase)
        for token in tokens:
            if token == AND.strip():
                parts = phrase.split(AND, 1)
                return "AND(%s, %s)" % (make_triples_from_phrase(parts[0]), make_triples_from_phrase(parts[1]))
            elif token == THAT.strip():
                parts = phrase.split(THAT, 1)
                return "AND(%s, %s)" % (make_triples_from_phrase(parts[0]), make_triples_from_phrase(parts[1]))
            elif token == OR.strip():
                parts = phrase.split(OR, 1)
                return "OR(%s, %s)" % (make_triples_from_phrase(parts[0]), make_triples_from_phrase(parts[1]))
    else:
        return make_one_triple(phrase)


def make_conjs(sentences):
    """
    Makes a conjunction from sentences.
    """
    conjs = ''

    for sentence in sentences:
        current_triple = make_one_triple(sentence)
        if current_triple is not None:
            conjs += str(current_triple)
            # Add a comma if it's not the last one.
            if sentences.index(sentence) != len(sentences) - 1:
                conjs += ', '
    return conjs


def get_subject(doc):
    for token in doc:
        if("subj" in token.dep_):
            return token.text
    
    return ""

def get_object_phrase(doc): # Unused
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i
            return doc[start:end]

def get_object(doc):
    for token in doc:
        if("dobj" in token.dep_):
            return token.text
    
    for token in doc:
        if("iobj" in token.dep_):
            return token.text

    for token in doc:
        if("pobj" in token.dep_):
            return token.text
        
    for token in doc:
        if("acomp" in token.dep_):
            return token.text
    
    for token in doc:
        if("xcomp" in token.dep_):
            return token.text
    return ""

def get_root(doc):
    for token in doc:
        if("ROOT" in token.dep_):
            return token.lemma_ # Lemmatized version of root is returned
    return ""



def make_one_triple(sentence: str) -> str:
    """
    Makes a single triple, that should be returned as a string.
    """
    neg = False
    relation = 'isA'
    obj = None
    if 'not' in sentence or 'never' in sentence:
        neg = True

    # tokens = word_tokenize(sentence)
    # TODO: 131You, numbers at start of word need to be trimmed
    """
    Trend for missing obj is usually a verb in infinitive form, for example:
    find out if you [are eligible] to renew. Here, [are eligible] should be the obj, but isn't recognized
    ideal triple would be something like (self, find, eligible)

    Incorrect obj:
    If the problem is safety-related, you must have the problem fixed immediately
    IF (problem, isA, be), THEN (self, hasA, problem)

    Likely spacy doesn't recognize this safety-related as one word indirect obj
    and self, hasA, problem should instead be (self, fix, problem). But this is a more complex case.


    If you really need to idle, shift to neutral, so the engine is not working against your brake and consuming more fuel
    IF (self, isA, need), THEN AND((engine, shift, neutral), (self, consume, fuel))
    should be IF (self, need, idle), THEN AND((engine, shift, neutral), (self, consume, fuel))
    ^^The temporary fix causes this one

    * Test for first 10 pages (Rules of the Road)
    * 


    """
    doc = nlp(sentence)
    
    try:
        subject = get_subject(doc)
        relation = get_root(doc)
        object = get_object(doc)

        if(subj_is_self(doc) or subject == ""):
            subject = "self"
        
        if(object == ""): # TEMPORARY FIX, find general case
            object = relation
            relation = "be"

        if(relation == "have"):
            relation = "hasA"
        elif(relation == "be"):
            relation = "isA"

        # Verb before subject errors.

        if neg:
            return f"NOT({subject}, {relation}, {object})"
        else:
            return f"({subject}, {relation}, {object})"
    except TypeError:
        logging.debug("Could not make a triple for text %s" % sentence)
    except IndexError:
        logging.debug("Sentence: %s is blank" % sentence)



def subj_is_self(doc) -> bool:
    if(get_subject(doc).lower() == "you" or get_subject(doc).lower() == "your"):
        return True
    return False


def parse_manual(state: str='MA', rule_file: str = ""):
    rules = read_manual(state, rule_file=rule_file)
    # for rule in rules:
        # print(rule)


if __name__ == "__main__":
    
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', '--verbose', action='store_true')
    parser.add_argument('--state', nargs='?', default='MA',
                        help='Name of the state to parse.  Options are CA (California) and MA (Massachusetts) the default.')
    parser.add_argument('--f', '--file', action='store_true',
                        help='Whether to write the rules (in natural language) to file or not.')

    args = parser.parse_args()
    if args.v:  # Set verbose messages if you want them.
        logging.getLogger().setLevel(logging.DEBUG)

    state = 'CA' if args.state.startswith('C') or args.state.startswith('c') else 'MA'
    # TODO: Add an option for writing out to file.
    parse_manual(state)


def high_level():
    if args.f:
        parse_manual(state, rule_file='rules_%s.txt'%args.state)
    else:
        parse_manual(state)
