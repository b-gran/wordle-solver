import json
import typing
from enum import Enum
import collections
from dataclasses import dataclass
from tqdm import tqdm
import itertools

from multiprocessing import Pool, cpu_count

top_5k_serialized = json.load(open('top_5k_words.json'))
dictionary = [e['word'] for e in top_5k_serialized]

alphabet = list('abcdefghijklmnopqrstuvwxyz')


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
    clue = []

    elusive = collections.defaultdict(int)
    for i, c in enumerate(solution):
        if guess[i] != c:
            elusive[c] += 1

    for i, guess_char in enumerate(guess):
        if solution[i] == guess_char:
            clue.append(LetterClue(type=Clue.PRESENT_CORRECT_LOCATION, character=guess_char, position=i))
        elif elusive[guess_char] > 0:
            elusive[guess_char] -= 1
            clue.append(LetterClue(type=Clue.PRESENT_INCORRECT_LOCATION, character=guess_char, position=i))
        else:
            clue.append(LetterClue(type=Clue.NOT_PRESENT, character=guess_char, position=i))

    return clue


@dataclass
class MetaClue:
    character: str

    fixed_positions: typing.Set[int]
    impossible_positions: typing.Set[int]
    lower_bound: int
    upper_bound: int


ALL_POSITIONS = {0, 1, 2, 3, 4}


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
                    mc.impossible_positions = ALL_POSITIONS.difference(mc.fixed_positions)

            # if we got at least one clue, we can update the lower bound on the
            # number of occurrences in the solution.
            elif clues[char] > 0 and clues[char] > mc.lower_bound:
                mc.lower_bound = clues[char]

            # if we got a not_present and no clues, we know
            # the character doesn't appear in the solution.
            if nps[char] > 0 and clues[char] == 0:
                mc.lower_bound = 0
                mc.upper_bound = 0
                mc.impossible_positions = ALL_POSITIONS

        sum_of_lower_bounds = 0
        all_fixed_positions = set()
        for mc in metaclues.values():
            sum_of_lower_bounds += mc.lower_bound
            all_fixed_positions.update(mc.fixed_positions)

        for mc in metaclues.values():
            mc.upper_bound = min(mc.upper_bound, 5 - (sum_of_lower_bounds - mc.lower_bound))
            if mc.upper_bound == 0:
                mc.impossible_positions = ALL_POSITIONS
            else:
                mc.impossible_positions.update(all_fixed_positions.difference(mc.fixed_positions))

    return metaclues


def filter_word(word: str, metaclues: typing.Dict[str, MetaClue], required_letters: typing.Set[str]) -> int:
    counts = collections.defaultdict(int)

    # for i, char in enumerate(word):
    for i in range(5):
        char = word[i]
        mc = metaclues[char]
        counts[char] += 1
        if counts[char] > mc.upper_bound:
            return 0

        if i in mc.impossible_positions:
            return 0

    for required_letter in required_letters:
        if counts[required_letter] < metaclues[required_letter].lower_bound:
            return 0

    return 1


def filter_impossible_words(dictionary: typing.List[str], metaclues: typing.Dict[str, MetaClue]) -> typing.List[str]:
    required_letters = set()
    for mc in metaclues.values():
        if mc.lower_bound > 0:
            required_letters.add(mc.character)

    return list(filter(lambda w: filter_word(w, metaclues, required_letters), dictionary))


def count_impossible_words(dictionary: typing.List[str], metaclues: typing.Dict[str, MetaClue]) -> int:
    required_letters = set()
    for mc in metaclues.values():
        if mc.lower_bound > 0:
            required_letters.add(mc.character)

    total = 0
    for w in dictionary:
        # if filter_word(w, metaclues, required_letters):
        #     total += 1
        total += filter_word(w, metaclues, required_letters)
    return total


def print_mc(mc: MetaClue):
    fp = "({})".format(''.join([str(d) if d in mc.fixed_positions else ' ' for d in range(5)]))
    ip = "({})".format(''.join([str(d) if d in mc.impossible_positions else ' ' for d in range(5)]))
    present = '✅' if mc.lower_bound > 0 else '❌' if mc.upper_bound == 0 else ' '
    return "{} {} {} [{}, {}] {}".format(mc.character, fp, ip, mc.lower_bound, mc.upper_bound, present)


def p(s):
    print('    fix     imp')
    for x in s.values():
        print(print_mc(x))


class Node:
    character: str
    children: typing.Dict[str, 'Node']
    prefix_counts: typing.DefaultDict[str, int]
    prefix: str

    def __init__(self, character: str, parent_prefix_counts: typing.DefaultDict[str, int], parent_prefix: str):
        self.character = character
        self.children = dict()
        self.prefix_counts = parent_prefix_counts.copy()
        self.prefix_counts[character] += 1
        self.prefix = parent_prefix + character


def make_trie(dictionary: typing.List[str]) -> Node:
    root = Node('', collections.defaultdict(int), '')

    for word in dictionary:
        curr = root
        for char in word:
            if char not in curr.children:
                curr.children[char] = Node(char, curr.prefix_counts, curr.prefix)
            curr = curr.children[char]

    return root


dictionary_trie = make_trie(dictionary)


def filter_dict_trie(dict_root: Node, metaclues: typing.Dict[str, MetaClue]) -> int:
    required_letters = set()
    for mc in metaclues.values():
        if mc.lower_bound > 0:
            required_letters.add(mc.character)

    depth = 0
    frontier = [dict_root]
    while frontier and depth < 5:
        frontier = [
            n.children[char] for n in frontier for char in n.children if
            n.prefix_counts[char] <= metaclues[char].upper_bound and depth not in metaclues[char].impossible_positions
        ]

        depth += 1

    num_valid = 0
    for n in frontier:
        valid = True
        for required_letter in required_letters:
            if n.prefix_counts[required_letter] < metaclues[required_letter].lower_bound:
                valid = False
                break

        if valid:
            num_valid += 1

    return num_valid


# ## Example
# c1 = get_clues('clamp', 'conch')
# s1 = summarize_clues([c1])
# print(len(filter_impossible_words(dictionary, s1)))
# print(filter_impossible_words(dictionary, s1))
# p(s1)
# print(filter_dict_trie(dictionary_trie, s1))
# print('=====')
# c2 = get_clues('clamp', 'occur')
# s2 = summarize_clues([c1, c2])
# print(len(filter_impossible_words(dictionary, s2)))
# print(filter_impossible_words(dictionary, s2))
# p(s2)
# print(filter_dict_trie(dictionary_trie, s2))
# print('=====')
# c3 = get_clues('clamp', 'lathe')
# s3 = summarize_clues([c1, c2, c3])
# print(len(filter_impossible_words(dictionary, s3)))
# print(filter_impossible_words(dictionary, s3))
# p(s3)
# print(filter_dict_trie(dictionary_trie, s3))
# print('=====')
# c4 = get_clues('clamp', 'lamps')
# s4 = summarize_clues([c1, c2, c3, c4])
# print(len(filter_impossible_words(dictionary, s4)))
# print(filter_impossible_words(dictionary, s4))
# p(s4)
# print(filter_dict_trie(dictionary_trie, s4))

# ## Example 2
# c1 = get_clues('which', 'twice')
# s1 = summarize_clues([c1])
# print(len(filter_impossible_words(dictionary, s1)))
# print(filter_impossible_words(dictionary, s1))
# p(s1)
# filter_dict_trie(dictionary_trie, s1)

deltas = collections.defaultdict(int)

dict_length = len(dictionary)


def get_delta(inp: typing.Tuple[str, str]) -> typing.Tuple[int, str]:
    solution, guess = inp
    clues = get_clues(solution, guess)
    sc = summarize_clues([clues])
    delta = dict_length - filter_dict_trie(dictionary_trie, sc)
    return delta, guess


def solve(clues: typing.List[WordClue]):
    initial_sc = summarize_clues(clues)
    filtered_dict = filter_impossible_words(dictionary, initial_sc)

    # possibilities = list(itertools.product(filtered_dict, dictionary))
    possibilities = list(itertools.product(filtered_dict, filtered_dict))

    try:
        workers = cpu_count()
    except NotImplementedError:
        print('failed to get workers')
        workers = 1
    print('using {} workers'.format(workers))

    # with tqdm(total=len(dictionary) * len(filtered_dict)) as pbar:
    with tqdm(total=len(filtered_dict) * len(filtered_dict)) as pbar:
        pool = Pool(processes=workers)
        result = pool.imap(get_delta, possibilities)
        for r in result:
            delta, guess = r
            deltas[guess] += delta
            pbar.update(1)

    filtered_dict.sort(key=lambda w: -deltas[w])
    # dictionary.sort(key=lambda w: -deltas[w])
    serialized_deltas = [{
        'word': w,
        'deltas': deltas[w]
    } for w in filtered_dict]
    # } for w in dictionary]

    out_file = open('deltas.json', 'w')
    json.dump(serialized_deltas, out_file)

    # print('\n'.join(list(map(lambda w: '{} {}'.format(w, deltas[w]), dictionary))))
    print('\n'.join(list(map(lambda w: '{} {}'.format(w, deltas[w]), filtered_dict))))


def from_wordle(guess: str, clue: typing.List[Clue]) -> WordClue:
    # @dataclass
    # class LetterClue:
    #     type: Clue
    #     character: str
    #     position: int
    result = []
    for i, char in enumerate(guess):
        result.append(LetterClue(
            type=clue[i],
            character=char,
            position=i,
        ))
    return result


clue1 = from_wordle('raise', [
    Clue.PRESENT_INCORRECT_LOCATION,
    Clue.PRESENT_INCORRECT_LOCATION,
    Clue.NOT_PRESENT,
    Clue.NOT_PRESENT,
    Clue.NOT_PRESENT
])
clue2 = from_wordle('croak', [
    Clue.NOT_PRESENT,
    Clue.PRESENT_CORRECT_LOCATION,
    Clue.NOT_PRESENT,
    Clue.PRESENT_INCORRECT_LOCATION,
    Clue.NOT_PRESENT
])

clues = [clue1]
sc = summarize_clues(clues)
print(filter_impossible_words(dictionary, sc))
p(sc)
filter_dict_trie(dictionary_trie, sc)

solve(clues)
