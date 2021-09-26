import kenlm
import os
from weakref import WeakValueDictionary
from interactive_visualization.graph_utils import Graph, Arc

class BaseToken:
    def __init__(self, parent, label, score, scorer_state):
        self.parent = parent
        self.label = label
        self.score = score
        self.scorer_state = scorer_state
        self.children = WeakValueDictionary()

    def prolong(self, label, model):
        if label == self.label:
            return self
        if label == '_':
            if label in self.children:
                return self.children[label]
            new_token = BaseToken(self, label, self.score, self.scorer_state)
            self.children[label] = new_token
            return new_token

        if label in self.children:
            return self.children[label]
        else:
            if label == ' ':
                last_word = self.backtrace_word()
                new_state = kenlm.State()
                lm_score = model.BaseScore(self.scorer_state, last_word, new_state)
                new_token = BaseToken(self, label, lm_score + self.score, new_state)
            else:
                new_token = BaseToken(self, label, self.score, self.scorer_state)
            self.children[label] = new_token
            return new_token

    def backtrace(self):
        cur = self
        result = []
        prev = '_'
        while cur is not None:
            if cur.label is not None:
                if cur.label != prev and cur.label != '_':
                    prev = cur.label
                    result.append(cur.label)
            cur = cur.parent

        return ''.join(reversed(result))

    def backtrace_word(self):
        cur = self
        result = []
        prev = '_'
        while cur is not None and cur.label != ' ':
            if cur.label is not None:
                if cur.label != prev and cur.label != '_':
                    prev = cur.label
                    result.append(cur.label)
            cur = cur.parent

        return ''.join(reversed(result))

    def __str__(self):
        root_str = "root" if self.parent is None else "Non-root"
        return f"Token: {self.label} {root_str} {self.score}"

    def __repr__(self):
        return self.__str__
    
    
def ctc_merge(s):
    prev = '_'
    result = []
    for c in s:
        if c == prev or c == '_':
            continue
        result.append(c)
        
    return ''.join(result)

from queue import Queue
# Для визуализации процесса поиска гипотез
def hypothesis_tree(tokens, compress = False):
    arcs = []
    roots = set()
    states = 0
    state_num = dict()
    for token in tokens:
        while token is not None and token.parent is not None:
            if token not in state_num:
                state_num[token] = states
                states += 1
                token = token.parent
            else:
                break

        if token is not None and token not in state_num:
            roots.add(token)
            state_num[token] = states
            states += 1
        
    q = Queue()
    for token in roots:
        q.put(token)
        
        
    adj_list = dict()
    while not q.empty():
        token = q.get()
        for label, child in token.children.items():
            if state_num[token] not in adj_list:
                adj_list[state_num[token]] = []
            adj_list[state_num[token]].append((state_num[child], label))
            q.put(child)
            
    for v, adj in adj_list.items():
        for i, (u, word) in enumerate(adj):
            if len(adj_list[u]) == 1:
                word = word + adj_list[u][0][1]
                adj[i][0], adj_list[u] = adj_list[u][0], word
                
    for v, adj in adj_list.items():
        for u, word in adj:
            arcs.append(v, u, word)
        
    tokens_set = {state_num[token] for token in tokens}

    result = Graph(arcs)
    for i, node in result.nodes.items():
        node.SetShape('point')
        if i in tokens_set:
            node.SetColor('red')
    return result.Visualize()

class ScoredToken:
    def __init__(self, token: BaseToken, score):
        self.token = token
        self.score = score

    def total_score(self):
        return self.score + self.token.score

    def consume_label(self, label, acoustic_score, model):
        return ScoredToken(self.token.prolong(label, model), self.score + acoustic_score)

    def __str__(self):
        return f"ScoredToken: {self.token} {self.score}"

    def __repr__(self):
        return self.__str__()

class CTCBeamSearchDecoder:
    def __init__(self,
                 kenlm_model,
                 alphabet,
                 beam_size = 50,
                 loglike_gap = 5.0,
                 acoustic_loglike_gap = 5,
                 detach_fixed_part = False,
                 save_trees = False,
                 show_fixed_part = False):
        self.kenlm_model = kenlm_model
        self.alphabet = alphabet
        self.beam_size = beam_size
        self.loglike_gap = loglike_gap
        self.acoustic_loglike_gap = acoustic_loglike_gap
        self.detach_fixed_part = detach_fixed_part
        self.save_trees = save_trees
        self.show_fixed_part = show_fixed_part
        if save_trees:
            self.trees = []

        self.reset()

    def reset(self):
        state = kenlm.State()
        self.kenlm_model.BeginSentenceWrite(state)
        self.tokens = [ScoredToken(BaseToken(None, None, 0, state), 0)]
        self.fixed_part = BaseToken(None, None, 0, None)
        self.num_frames = 0

    def best_hyp(self):
        best_token = max(self.tokens, key=lambda x: x.total_score())

        return (self.fixed_part.backtrace() if self.fixed_part is not None else "") + best_token.token.backtrace()

    def choose_labels(self, loglikes):
        max_loglike = max(loglikes)
        return [i for i, x in enumerate(loglikes) if x > max_loglike - self.acoustic_loglike_gap]

    def prune(self):
        new_tokens = list(sorted(self.tokens, key= lambda x: -x.total_score()))[:self.beam_size]
        best_score = new_tokens[0].total_score()
        self.tokens = [x for x in new_tokens if x.total_score() >= best_score - self.acoustic_loglike_gap]
        
    def prune_state(self):
        best_states = dict()
        for token in self.tokens:
            word = token.token.backtrace_word()
            if word not in best_states or best_states[word].score < token.score:
                best_states[word] = token
                
        self.tokens = list(best_states.values())

    def process_frame(self, loglikes):
        self.num_frames += 1
        labels = self.choose_labels(loglikes)

        new_tokens = []
        for token in self.tokens:
            for label in labels:
                new_tokens.append(token.consume_label(self.alphabet[label], loglikes[label], self.kenlm_model))

        self.tokens = new_tokens
        self.prune()
        
        #if self.num_frames % 4 == 0:
        #    self.prune_state()
        
        if self.detach_fixed_part:
            self.detatch()

        if self.save_trees:
            self.trees.append(self.tree())

    def detatch(self):
        active_tokens_set = {x.token for x in self.tokens}
        root = self.tokens[0].token
        while root.parent is not None:
            root = root.parent


        branch = root
        while len(branch.children) == 1 and branch not in active_tokens_set:
            for next_token in branch.children.values():
                branch = next_token

        if branch.parent is not None:
            self.fixed_part = self.fixed_part.parent
            root.parent = self.fixed_part
            if self.fixed_part is not None:
                self.fixed_part.children[root.label] = root
            branch.parent.children.clear()
            self.fixed_part = BaseToken(branch.parent, branch.label, 0, None)
            branch.parent.children[branch.label] = self.fixed_part
            branch.parent = None

    def tree(self):
        tokens = []
        if self.fixed_part.parent is not None and self.show_fixed_part:
            tokens.append(self.fixed_part)
        for token in self.tokens:
            tokens.append(token.token)
        return hypothesis_tree(tokens)

    def fixed_part_tree(self):
        return hypothesis_tree([self.fixed_part])

    def fixed_hyp(self):
        return self.fixed_part.backtrace()
