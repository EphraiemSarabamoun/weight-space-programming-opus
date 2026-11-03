The network is saved and complete. I never ran it or tested it, as the rules require, so the expected strength below comes from the design alone.

**How the layers are used.** Layers 1–5 each carry exact copies of the "own stone" and "opponent stone" planes, so every layer can still read the raw board. The rest of each layer computes tactical features. Every unit is written by hand as a relu of an integer linear form.

**Liberty analysis (layers 1–5), done for both colours:**
- **Layer 1:** for each stone, a count that is zero unless the stone has at least two empty neighbours.
- **Layers 2–4:** this "has spare liberties" signal spreads along same-coloured stones out to radius 3. A large negative bias blocks it from crossing to the other colour or to empty points.
- **Layer 4 check:** the liberties of the stone's direct same-coloured neighbours must be zero.
- **Layer 5 output:** an atari flag per stone, meaning "this string has only one liberty."
- **Accuracy:** the flag is exact for strings of one or two stones. For larger strings it is a radius-3 approximation that can also flag some strings with two liberties.

**Territory model (layers 4–6).**
- **Influence:** each point gets a score from nearby stones, weighted by distance (1, 0.7, 0.5, …), counting ours as plus and the opponent's as minus.
- **Contested points:** layer 5 turns this into a "tent" signal that is highest where influence is near zero, and only on empty points.
- **Move value:** layer 6 adds up the tent signal around each move with the same distance weights. So the policy plays where it can flip the most contested area. On an empty board that is the centre; later it is the frontier between the two sides.

**Final move score (scaled ×3 into logits):**
- **Captures:** +25 for each adjacent opponent stone whose string is in atari. +4 or +8 more for pressing a single stone down to two liberties or one.
- **Escapes:** +10 to +30 for extending a string in atari, but only if the move leaves at least two liberties.
- **Self-atari:** −10 or −20 when the new stone would have at most one liberty and the move captures nothing.
- **Own eyes:** −60 for filling a point whose neighbours are all our own stones.
- **Dead stones in our area:** a bonus for playing next to opponent stones that sit inside our area, so they get captured rather than spoiling our territory.
- **Invasions:** a small bonus for moves deep in opponent territory, which only matter late in the game.
- **Other terms:** a small penalty for first- and second-line moves and a slight preference against filling our own area.

**Passing.** The pass logit is a constant. It beats filling our own territory, eye fills and self-ataris, but loses to any contested point, capture or invasion.

**Why I expect it to win.** The trained opponents beat a random player only 79–98% of the time, so they are weak. Against them, the policy should:
- keep its own eyes;
- capture stones left in atari, both singles and larger groups;
- rescue its own strings when the escape really adds liberties;
- avoid throwing stones away into atari;
- clear dead stones out of its own area, which matters under area scoring;
- otherwise spend each move on the most contested ground.

These are the mistakes weak policies make, and this one makes them rarely and almost deterministically.

**Known weaknesses:**
- It has no ladder reading, so it can extend a string in atari into a ladder that fails.
- It has no idea of whole-group life and death.
- Large strings are judged only approximately, so some "captures" are really just ataris.