import json
import typing
from enum import Enum
import collections
from dataclasses import dataclass
from tqdm import tqdm

top_5k_serialized = json.load(open('top_5k_words.json'))
dictionary = [e['word'] for e in top_5k_serialized]

alphabet = list('abcdefghijklmnopqrstuvwxyz')

freq_by_char = {}
for c in alphabet:
    freq_by_char[c] = 0
for w in dictionary:
    for c in w:
        freq_by_char[c] = freq_by_char[c] + 1
x = list(freq_by_char.items())
x.sort(key=lambda i: i[1])


def char_freq(w: str) -> typing.Dict[str, int]:
    r = collections.defaultdict(int)
    for c in w:
        r[c] = r[c] + 1
    return r


class Clue(Enum):
    NOT_PRESENT = 0
    PRESENT_INCORRECT_LOCATION = 1
    PRESENT_CORRECT_LOCATION = 2


@dataclass
class LetterClue:
    type: Clue
    character: str
    position: int


WordClue = typing.List[LetterClue]


# from https://github.com/lynn/hello-wordl/blob/00ef8569149bd998f5a992853172a1a30b4914b2/src/clue.ts#L12
def get_clues(solution: str, guess: str) -> WordClue:
    clue = [0, 0, 0, 0, 0]

    elusive = collections.defaultdict(int)
    for i, c in enumerate(solution):
        if guess[i] != c:
            elusive[c] += 1

    for i, guess_char in enumerate(guess):
        if solution[i] == guess_char:
            # clue.append(LetterClue(type=Clue.PRESENT_CORRECT_LOCATION, character=guess_char, position=i))
            clue[i] = LetterClue(type=Clue.PRESENT_CORRECT_LOCATION, character=guess_char, position=i)
        elif elusive[guess_char] > 0:
            elusive[guess_char] -= 1
            # clue.append(LetterClue(type=Clue.PRESENT_INCORRECT_LOCATION, character=guess_char, position=i))
            clue[i] = LetterClue(type=Clue.PRESENT_INCORRECT_LOCATION, character=guess_char, position=i)
        else:
            # clue.append(LetterClue(type=Clue.NOT_PRESENT, character=guess_char, position=i))
            clue[i] = LetterClue(type=Clue.NOT_PRESENT, character=guess_char, position=i)

    return clue


@dataclass
class MetaClue:
    character: str

    fixed_positions: typing.Set[int]
    impossible_positions: typing.Set[int]
    lower_bound: int
    upper_bound: int


def summarize_clues(clues: typing.List[WordClue]) -> typing.Dict[str, MetaClue]:
    metaclues = {}
    for c in alphabet:
        metaclues[c] = MetaClue(
            character=c,
            fixed_positions=set(),
            impossible_positions=set(),
            lower_bound=0,
            upper_bound=5
        )

    for clue in clues:
        freqs = collections.defaultdict(int)
        clues = collections.defaultdict(int)
        nps = collections.defaultdict(int)
        chars = set()

        for lc in clue:
            chars.add(lc.character)
            freqs[lc.character] += 1
            if lc.type != Clue.NOT_PRESENT:
                clues[lc.character] += 1
            else:
                nps[lc.character] += 1

            if lc.type == Clue.PRESENT_CORRECT_LOCATION:
                metaclues[lc.character].fixed_positions.add(lc.position)
            else:
                metaclues[lc.character].impossible_positions.add(lc.position)

        for char in chars:
            mc = metaclues[char]
            # if we didn't receive a clue for each occurrence of the character,
            # we can determine how many times the character appears in the solution.
            if freqs[char] > clues[char]:
                mc.lower_bound = clues[char]
                mc.upper_bound = clues[char]

                # we already know all positions for the character, so we can mark the rest as impossible
                if len(mc.fixed_positions) == clues[char]:
                    mc.impossible_positions = {0, 1, 2, 3, 4}.difference(mc.fixed_positions)

            # if we got at least one clue, we can update the lower bound on the
            # number of occurrences in the solution.
            elif clues[char] > 0 and clues[char] > mc.lower_bound:
                mc.lower_bound = clues[char]

            # if we got a not_present and no clues, we know
            # the character doesn't appear in the solution.
            if nps[char] > 0 and clues[char] == 0:
                mc.lower_bound = 0
                mc.upper_bound = 0
                mc.impossible_positions = {0, 1, 2, 3, 4}

        sum_of_lower_bounds = 0
        all_fixed_positions = set()
        for mc in metaclues.values():
            sum_of_lower_bounds += mc.lower_bound
            all_fixed_positions.update(mc.fixed_positions)

        for mc in metaclues.values():
            mc.upper_bound = min(mc.upper_bound, 5-(sum_of_lower_bounds-mc.lower_bound))
            if mc.upper_bound == 0:
                mc.impossible_positions = {0, 1, 2, 3, 4}
            else:
                mc.impossible_positions.update(all_fixed_positions.difference(mc.fixed_positions))

    return metaclues


def filter_word(word: str, metaclues: typing.Dict[str, MetaClue]) -> bool:
    counts = collections.defaultdict(int)
    for i, char in enumerate(word):
        mc = metaclues[char]
        counts[char] += 1
        if counts[char] > mc.upper_bound:
            return False

        if i in mc.impossible_positions:
            return False

    for mc in metaclues.values():
        if counts[mc.character] < mc.lower_bound:
            return False

    return True


def filter_impossible_words(dictionary: typing.List[str], metaclues: typing.Dict[str, MetaClue]) -> typing.List[str]:
    return list(filter(lambda w: filter_word(w, metaclues), dictionary))


def print_mc(mc: MetaClue):
    fp = "({})".format(''.join([str(d) if d in mc.fixed_positions else ' ' for d in range(5)]))
    ip = "({})".format(''.join([str(d) if d in mc.impossible_positions else ' ' for d in range(5)]))
    present = '✅' if mc.lower_bound > 0 else '❌' if mc.upper_bound == 0 else ' '
    return "{} {} {} [{}, {}] {}".format(mc.character, fp, ip, mc.lower_bound, mc.upper_bound, present)


def p(s):
    print('    fix     imp')
    for x in s.values():
        print(print_mc(x))


# ## Example
# c1 = get_clues('clamp', 'conch')
# s1 = summarize_clues([c1])
# print(len(filter_impossible_words(dictionary, s1)))
# print(filter_impossible_words(dictionary, s1))
# p(s1)
# print('=====')
# c2 = get_clues('clamp', 'occur')
# s2 = summarize_clues([c1, c2])
# print(len(filter_impossible_words(dictionary, s2)))
# print(filter_impossible_words(dictionary, s2))
# p(s2)
# print('=====')
# c3 = get_clues('clamp', 'lathe')
# s3 = summarize_clues([c1, c2, c3])
# print(len(filter_impossible_words(dictionary, s3)))
# print(filter_impossible_words(dictionary, s3))
# p(s3)
# print('=====')
# c4 = get_clues('clamp', 'lamps')
# s4 = summarize_clues([c1, c2, c3, c4])
# print(len(filter_impossible_words(dictionary, s4)))
# print(filter_impossible_words(dictionary, s4))
# p(s4)

deltas = collections.defaultdict(int)

# with tqdm(total=n * m) as pbar:
#     for i1 in tqdm(range(n)):
#         for i2 in tqdm(range(m)):
#             # do something, e.g. sleep
#             time.sleep(0.01)
#             pbar.update(1)

dict_length = len(dictionary)

with tqdm(total=len(dictionary)**2) as pbar:
    for solution in dictionary:
        for guess in dictionary:
            clues = get_clues(solution, guess)
            sc = summarize_clues([clues])
            delta = dict_length - len(filter_impossible_words(dictionary, sc))
            deltas[guess] += delta
            pbar.update(1)

dictionary.sort(key=lambda w: -deltas[w])

print('\n'.join(list(map(lambda w: '{} {}'.format(w, deltas[w]), dictionary[:100]))))
