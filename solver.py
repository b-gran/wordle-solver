import json
import typing
from enum import Enum
import collections
from dataclasses import dataclass
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np
import argparse


def get_normalized_freq_by_word(fbw: typing.Union[typing.Dict[str, int], typing.DefaultDict[str, int]]) -> typing.DefaultDict[str, float]:
    min_freq = min(fbw.values())
    result = collections.defaultdict(float)
    result.update({
        k: fbw[k] / min_freq for k in fbw
    })
    return result


invalid_words = set(open("invalid_words.txt").read().splitlines())
top_5k_serialized = json.load(open('top_5k_words.json'))
freq_by_word = collections.defaultdict(int)
freq_by_word.update({
    f['word']: f['frequency'] for f in top_5k_serialized
})
normalized_freq_by_word = get_normalized_freq_by_word(freq_by_word)
frequencies = [e['frequency'] for e in top_5k_serialized]

freq_p30 = np.quantile(frequencies, 0.3)
freq_p90 = np.quantile(frequencies, 0.9)

# wordle
solutions = list(open('solutions.txt').read().splitlines())
accepted = list(open('accepted.txt').read().splitlines())

# hello wordl
# solutions = list(open('solutions_hello_wordl.txt').read().splitlines())
# accepted = list(open('accepted_hello_wordl.txt').read().splitlines())


def get_solution_probability_adjustment(frequency: typing.Union[int, float]) -> int:
    if frequency >= freq_p90:
        return 2
    elif frequency >= freq_p30:
        return 4
    else:
        return 1


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


dictionary_trie = make_trie(solutions)


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


deltas = collections.defaultdict(int)


def get_delta(inp: typing.Tuple[str, str, int, Node]) -> typing.Tuple[float, str]:
    """
        get_delta(solution: str, guess: str, dict_size: int, dict_trie: Node]) -> typing.Tuple[int, str]
    """
    solution, guess, dict_size, dict_trie = inp
    clues = get_clues(solution, guess)
    sc = summarize_clues([clues])

    # non-normalized - just checks how many solutions are removed with this guess
    # delta = dict_size - filter_dict_trie(dict_trie, sc)

    # normalized - what percentage of all solutions were removed
    delta = (dict_size - filter_dict_trie(dict_trie, sc)) * 1.0 / dict_size

    return delta, guess


def solve(clues: typing.List[WordClue]):
    initial_sc = summarize_clues(clues)
    filtered_dict = filter_impossible_words(solutions, initial_sc)
    trie = make_trie(filtered_dict)

    solution_dict = filtered_dict.copy()
    guess_dict = accepted.copy()

    # TODO: this will only guess possible solutions
    # solution_dict = filtered_dict
    # guess_dict = filtered_dict

    possibilities = list(itertools.product(solution_dict, guess_dict, [len(filtered_dict)], [trie]))

    try:
        workers = cpu_count()
    except NotImplementedError:
        print('failed to get workers')
        workers = 1
    print('using {} workers'.format(workers))

    with tqdm(total=len(solution_dict) * len(guess_dict)) as pbar:
        pool = Pool(processes=workers)
        result = pool.imap(get_delta, possibilities)
        for r in result:
            delta, guess = r
            deltas[guess] += (delta/len(solution_dict))
            pbar.update(1)

    def score(guess: str) -> float:
        if guess in solution_dict:
            return deltas[guess] * get_solution_probability_adjustment(freq_by_word[guess])
        else:
            return deltas[guess]

    def choose() -> str:
        # only one option let, choose it!
        if len(solution_dict) == 1:
            return solution_dict[0]

        guess_dict.sort(key=lambda w: -score(w))

        # if we have only 2 solutions, just pick the best guess by score. tree pruning doesn't matter.
        if len(solution_dict) < 3:
            return guess_dict[0]

        # if we have more than 2 guesses, first check if there is an obvious "best" choice (defined by the top
        # choice being 2x larger than the second best). if so, choose this option and try to win the game.
        if score(guess_dict[0]) > 2*score(guess_dict[1]):
            print('Choosing best option by score')
            return guess_dict[0]

        # otherwise, default to a tree pruning strategy.
        print('Choosing best option by tree pruning ratio')
        guess_dict.sort(key=lambda w: -deltas[w])
        return guess_dict[0]

    # Exploring several ways to choose the guess
    # print('top 20 guesses (by frequency)')
    # guess_dict.sort(key=lambda w: -normalized_freq_by_word[w])
    # print('\n'.join(list(map(lambda w: '{} {}'.format(w, normalized_freq_by_word[w]), guess_dict[:20]))))
    # print('===========')
    #
    # print('top 20 guesses (by delta)')
    # guess_dict.sort(key=lambda w: -deltas[w])
    # print('\n'.join(list(map(
    #     lambda w: '{} {}'.format(w, deltas[w]), guess_dict[:20])
    # )))
    # print('===========')
    #
    # print('top 20 guesses (by quantile adjustment)')
    # guess_dict.sort(key=lambda w: -get_solution_probability_adjustment(freq_by_word[w]))
    # print('\n'.join(list(map(
    #     lambda w: '{} {}'.format(w, get_solution_probability_adjustment(freq_by_word[w])), guess_dict[:20])
    # )))
    # print('===========')
    #
    # print('top 20 guesses (by score)')
    # guess_dict.sort(key=lambda w: -score(w))
    # print('\n'.join(list(map(
    #     lambda w: '{} {}'.format(w, score(w)), guess_dict[:20])
    # )))
    # print('===========')

    guess_dict.sort(key=lambda w: -score(w))
    serialized_deltas = [{
        'word': w,
        'deltas': score(w)
    } for w in guess_dict]

    out_file = open('deltas.json', 'w')
    json.dump(serialized_deltas, out_file)

    best_option = choose()
    print('Best option:', best_option)


def from_wordle(guess: str, clue: typing.List[Clue]) -> WordClue:
    result = []
    for i, char in enumerate(guess):
        result.append(LetterClue(
            type=clue[i],
            character=char,
            position=i,
        ))
    return result


def parse_clue_cli(guess: str, clue: str) -> WordClue:
    assert len(guess) == 5, 'Guess {} must be 5 characters long'.format(guess)
    assert len(clue) == 5, 'Clue {} must be 5 characters long'.format(clue)

    result = []
    for i, char in enumerate(guess):
        c = Clue.NOT_PRESENT
        if clue[i].lower() == 'g':
            c = Clue.PRESENT_CORRECT_LOCATION
        elif clue[i].lower() == 'y':
            c = Clue.PRESENT_INCORRECT_LOCATION

        result.append(LetterClue(
            type=c,
            character=char,
            position=i,
        ))
    return result


parser = argparse.ArgumentParser(description='Solve wordle!')
parser.add_argument('--guess', action='append', help='A guess that you entered', required=True)
parser.add_argument('--clues', action='append', help='The clues you received for this guess', required=True)

# Execute the parse_args() method
args = parser.parse_args()

assert len(args.guess) == len(args.clues), 'Must provide the same number of clues as guesses'

clues = list(map(lambda i: parse_clue_cli(args.guess[i], args.clues[i]), range(len(args.guess))))

solve(clues)
