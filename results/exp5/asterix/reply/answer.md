# Hand-built Asterix policy: a vertical dodger that stays in column 5 and seeks treasure

## Strategy

The player never moves left or right, so it stays in its starting column 5. Every entity crosses its whole row, so every treasure eventually passes column 5. Every threat can be dodged by moving one row up or down.

Two facts about the game make this work:

- **One entity per row.** Each row holds at most one entity, so the trail cell in a row always belongs to that row's entity. A linear sum over one row can therefore read both where the entity is and which way it is moving.
- **Treasure rows are safe.** A row that holds a treasure cannot hold an enemy. Waiting in a row whose treasure is approaching is both safe and pays +1.

Each step the network compares three choices: stay, go up, go down. Each choice is scored by the danger in the row it leads to, plus the pull of approaching treasure in that direction.

- **Stay:**
  - safe row: logit about 4
  - safe row with approaching treasure: about 12
  - enemy 2 cells away and approaching: about -3
  - enemy 1 cell away and approaching: about -12
- **Move into a row:**
  - safe row: about 0
  - safe row, treasure in that direction: about 8, smaller for treasure further away
  - enemy 2 cells away: about -9
  - enemy 1 cell away or at column 5: about -16
- **Edge rows:**
  - "Up" from row 1 and "down" from row 8 are treated as staying in that row.
  - Rows 1 and 8 get a slightly lower stay value and a small push toward the centre.
- **Other actions:** left, right and fire get a fixed logit of -30.

## Hidden layer 1

| Units | What each computes | How |
|---|---|---|
| 0–7 | P_r: the player is in row r (rows 1–8) at column 5 | Single input with a sharp threshold |
| 8–15 | Dg_r: danger at column 5 in row r | Linear sum over enemy and trail cells of the row, then tanh |
| 16–23 | G_r: row r holds a treasure that has not yet passed column 5 | Treasure weight by column, minus trail weight by column, sharp threshold |
| 24–31 | Unused | — |

Details for the danger units (Dg_r):
- The row sum takes these levels:
  - enemy at column 5: 3
  - enemy 1 cell away and approaching (column 4 moving right, or 6 moving left): 2
  - 2 cells away and approaching: 1
  - 3 cells away: 0.5
  - enemy moving away, treasure, or empty row: 0 or less
- The output is tanh(2·(S−1.5)). This separates the 1-cell-away level strongly from the 2-cell level.

Details for the treasure units (G_r):
- The weights give about 1.5 when the treasure is approaching and 0.75 or less otherwise.
- This holds for every column and direction, including entities that just spawned and have no trail yet.
- An enemy's trail alone can never reach the threshold.

## Hidden layer 2

There are 24 units, in three groups, one unit per player row in each group:
- Here_r (units 0–7)
- Up_r (units 8–15)
- Down_r (units 16–23)

Gating by row:
- Each unit has weight +10 from P_r and a bias that includes -10.
- In the player's row the unit computes tanh(f).
  - Stay: f = constant − a·(danger in this row) + b·(treasure in this row).
  - Move: f = constant − a·(danger in the target row) + Σ w(k)·(treasure k rows away in that direction).
- In every other row the unit sits at exactly -1. The output biases cancel those constant -1 contributions.
- The treasure weights w(k) fall off with row distance k, so nearer treasure pulls harder.

## Output

- No-op = 12·ΣHere; up = 12·ΣUp − 4; down = 12·ΣDown − 4.
- Left, right and fire have a bias of -30.

```edits
# ===== Layer 1 =====
# P_r : player at (row r, col 5), units 0..7  (r = unit+1)
W1[90, 0] = 8
W1[150, 1] = 8
W1[210, 2] = 8
W1[270, 3] = 8
W1[330, 4] = 8
W1[390, 5] = 8
W1[450, 6] = 8
W1[510, 7] = 8
b1[0:8] = -4
# Dg_r : danger at column 5 in row r, units 8..15
# enemy cols 2..8 weights, trail cols 3..7 weights
W1[73:110:6, 8] = [1, 2, 5, 10, 5, 2, 1]
W1[82:107:6, 8] = [-1, -4, -6, -4, -1]
W1[133:170:6, 9] = [1, 2, 5, 10, 5, 2, 1]
W1[142:167:6, 9] = [-1, -4, -6, -4, -1]
W1[193:230:6, 10] = [1, 2, 5, 10, 5, 2, 1]
W1[202:227:6, 10] = [-1, -4, -6, -4, -1]
W1[253:290:6, 11] = [1, 2, 5, 10, 5, 2, 1]
W1[262:287:6, 11] = [-1, -4, -6, -4, -1]
W1[313:350:6, 12] = [1, 2, 5, 10, 5, 2, 1]
W1[322:347:6, 12] = [-1, -4, -6, -4, -1]
W1[373:410:6, 13] = [1, 2, 5, 10, 5, 2, 1]
W1[382:407:6, 13] = [-1, -4, -6, -4, -1]
W1[433:470:6, 14] = [1, 2, 5, 10, 5, 2, 1]
W1[442:467:6, 14] = [-1, -4, -6, -4, -1]
W1[493:530:6, 15] = [1, 2, 5, 10, 5, 2, 1]
W1[502:527:6, 15] = [-1, -4, -6, -4, -1]
b1[8:16] = -3
# G_r : treasure in row r approaching column 5, units 16..23
W1[62:120:6, 16] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[64:120:6, 16] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[122:180:6, 17] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[124:180:6, 17] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[182:240:6, 18] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[184:240:6, 18] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[242:300:6, 19] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[244:300:6, 19] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[302:360:6, 20] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[304:360:6, 20] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[362:420:6, 21] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[364:420:6, 21] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[422:480:6, 22] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[424:480:6, 22] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
W1[482:540:6, 23] = [12, 8, 16, 24, 32, 40, 32, 24, 16, 12]
W1[484:540:6, 23] = [4, -6, -12, -20, -28, -36, -28, -20, -12, -4]
b1[16:24] = -8

# ===== Layer 2 =====  cols 0..7 Here_r, 8..15 Up_r, 16..23 Down_r
# gating by player row
W2[0, 0:24:8] = [10, 10, 10]
W2[1, 1:24:8] = [10, 10, 10]
W2[2, 2:24:8] = [10, 10, 10]
W2[3, 3:24:8] = [10, 10, 10]
W2[4, 4:24:8] = [10, 10, 10]
W2[5, 5:24:8] = [10, 10, 10]
W2[6, 6:24:8] = [10, 10, 10]
W2[7, 7:24:8] = [10, 10, 10]
# danger: own row for Here, row above for Up, row below for Down (edges = own row)
W2[8, 0] = -2.58
W2[8, 8] = -3.5
W2[8, 9] = -3.5
W2[9, 1] = -2.58
W2[9, 10] = -3.5
W2[9, 16] = -3.5
W2[10, 2] = -2.58
W2[10, 11] = -3.5
W2[10, 17] = -3.5
W2[11, 3] = -2.58
W2[11, 12] = -3.5
W2[11, 18] = -3.5
W2[12, 4] = -2.58
W2[12, 13] = -3.5
W2[12, 19] = -3.5
W2[13, 5] = -2.58
W2[13, 14] = -3.5
W2[13, 20] = -3.5
W2[14, 6] = -2.58
W2[14, 15] = -3.5
W2[14, 21] = -3.5
W2[15, 7] = -2.58
W2[15, 22] = -3.5
W2[15, 23] = -3.5
# treasure in own row -> stay
W2[16, 0] = 1.25
W2[17, 1] = 1.25
W2[18, 2] = 1.25
W2[19, 3] = 1.25
W2[20, 4] = 1.25
W2[21, 5] = 1.25
W2[22, 6] = 1.25
W2[23, 7] = 1.25
# treasure above -> Up (distance weighted)
W2[16:24, 8:16] = [[0, 1.25, 0.75, 0.5625, 0.475, 0.4125, 0.375, 0.35],
  [0, 0, 1.25, 0.75, 0.5625, 0.475, 0.4125, 0.375],
  [0, 0, 0, 1.25, 0.75, 0.5625, 0.475, 0.4125],
  [0, 0, 0, 0, 1.25, 0.75, 0.5625, 0.475],
  [0, 0, 0, 0, 0, 1.25, 0.75, 0.5625],
  [0, 0, 0, 0, 0, 0, 1.25, 0.75],
  [0, 0, 0, 0, 0, 0, 0, 1.25],
  [0, 0, 0, 0, 0, 0, 0, 0]]
# treasure below -> Down (distance weighted)
W2[16:24, 16:24] = [[0, 0, 0, 0, 0, 0, 0, 0],
  [1.25, 0, 0, 0, 0, 0, 0, 0],
  [0.75, 1.25, 0, 0, 0, 0, 0, 0],
  [0.5625, 0.75, 1.25, 0, 0, 0, 0, 0],
  [0.475, 0.5625, 0.75, 1.25, 0, 0, 0, 0],
  [0.4125, 0.475, 0.5625, 0.75, 1.25, 0, 0, 0],
  [0.375, 0.4125, 0.475, 0.5625, 0.75, 1.25, 0, 0],
  [0.35, 0.375, 0.4125, 0.475, 0.5625, 0.75, 1.25, 0]]
b2[0:8] = [-11.1491, -10.9701, -10.9701, -10.9701, -10.9701, -10.9701, -10.9701, -11.1491]
b2[8:16] = [-13.1355, -11.8855, -11.1355, -10.573, -10.098, -9.6855, -9.3105, -8.8645]
b2[16:24] = [-8.8645, -9.3105, -9.6855, -10.098, -10.573, -11.1355, -11.8855, -13.1355]

# ===== Output =====
W3[0:8, 0] = 12
W3[8:16, 2] = 12
W3[16:24, 4] = 12
b3[0:6] = [84, -30, 80, -30, 80, -30]
```