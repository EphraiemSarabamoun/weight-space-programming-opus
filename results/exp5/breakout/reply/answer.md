# Breakout policy network: hand-set weights for a catch-point predictor

## Design

**Strategy.** The ball's horizontal motion is a fold of a straight line. If the ball is at column `bx`, moves in direction `d` and has `s` steps left, it reaches column `F(bx + d·s)`. Here `F` reflects the unfolded position with period 20, and it reproduces the game's wall clamp, where the ball stays in an edge column for two steps. The paddle catches the ball if, after the paddle's own move, it sits on the ball's column or the column the ball is about to enter. These two cells form a **catch set** `{Lo, Hi}`. The policy moves right if the paddle is left of `Lo`, moves left if it is right of `Hi`, and does nothing otherwise.

The paddle therefore goes to the catch set early and waits there with no-op. That waiting gives slack against sticky actions: an overshoot by one cell usually still lands inside the catch set, and there is time to correct.

- **Descending ball:** `s = 8 − row`, which is exact because there are no bricks below row 3.
- **Ascending ball:** I assume the ball bounces off the next brick row: row 3 when the ball is at rows 4–8, the row just above at rows 2–3, and the ceiling at rows 0–1. With that assumption:
  - rows 4–8: `s = row + 1`
  - rows 2–3: `s = 9 − row`
  - row 1: `s = 10`
  - row 0: `s = 9`
- If the ball actually goes higher, the descending prediction takes over once the ball turns down, and there are still at least 5 moves left.

**Layer 1 (10 units)**
- **u0–u7, comparison units.** Each has a 100-cell lookup table on the ball channel giving that case's catch-set bound, a weight of −column on the paddle cells, and is scaled by 6. There is one unit for "Lo > paddle" (move right) and one for "Hi < paddle" (move left) in each of four cases: descending-right, descending-left, ascending-right, ascending-left.
- **u8, "descending" gate.** It computes `2·ball_row − 2·trail_row + 1`, with an extra −2 when the trail is in row 8. This counts brick or ceiling reflections (ball and trail in the same row) as descending, and the bounce off the paddle as ascending.
- **u9, "moving right" gate.** It computes `ball_col − trail_col`, with the ball's column-0 weight raised to 1. This makes the clamped states (0,0) and (9,9) give the true next direction. It also covers the reset state, where trail = ball.

**Layer 2 (8 units).** Each unit is an AND of one comparison unit with the matching gate polarities: `5·(u + ±D + ±R − 2)`.

**Output.** Right = OR of the four "move right" ANDs, giving logit 40. Left is built the same way. No-op has a constant logit of 20. This makes the policy close to deterministic: an error has probability of about e⁻²⁰ per step.

## Weights

```edits
# ---------- u0: Lo, descending-right (rows 0..8 of ball channel) ----------
W1[3:60:6, 0] = [8,9,8,7,6,5,4,3,2,1]
W1[63:120:6, 0] = [7,8,9,8,7,6,5,4,3,2]
W1[123:180:6, 0] = [6,7,8,9,8,7,6,5,4,3]
W1[183:240:6, 0] = [5,6,7,8,9,8,7,6,5,4]
W1[243:300:6, 0] = [4,5,6,7,8,9,8,7,6,5]
W1[303:360:6, 0] = [3,4,5,6,7,8,9,8,7,6]
W1[363:420:6, 0] = [2,3,4,5,6,7,8,9,8,7]
W1[423:480:6, 0] = [1,2,3,4,5,6,7,8,9,8]
W1[483:540:6, 0] = [0,1,2,3,4,5,6,7,8,9]
# ---------- u1: Hi, descending-right ----------
W1[3:60:6, 1] = [9,9,9,8,7,6,5,4,3,2]
W1[63:120:6, 1] = [8,9,9,9,8,7,6,5,4,3]
W1[123:180:6, 1] = [7,8,9,9,9,8,7,6,5,4]
W1[183:240:6, 1] = [6,7,8,9,9,9,8,7,6,5]
W1[243:300:6, 1] = [5,6,7,8,9,9,9,8,7,6]
W1[303:360:6, 1] = [4,5,6,7,8,9,9,9,8,7]
W1[363:420:6, 1] = [3,4,5,6,7,8,9,9,9,8]
W1[423:480:6, 1] = [2,3,4,5,6,7,8,9,9,9]
W1[483:540:6, 1] = [1,2,3,4,5,6,7,8,9,9]
# ---------- u2: Lo, descending-left ----------
W1[3:60:6, 2] = [7,6,5,4,3,2,1,0,0,0]
W1[63:120:6, 2] = [6,5,4,3,2,1,0,0,0,1]
W1[123:180:6, 2] = [5,4,3,2,1,0,0,0,1,2]
W1[183:240:6, 2] = [4,3,2,1,0,0,0,1,2,3]
W1[243:300:6, 2] = [3,2,1,0,0,0,1,2,3,4]
W1[303:360:6, 2] = [2,1,0,0,0,1,2,3,4,5]
W1[363:420:6, 2] = [1,0,0,0,1,2,3,4,5,6]
W1[423:480:6, 2] = [0,0,0,1,2,3,4,5,6,7]
W1[483:540:6, 2] = [0,0,1,2,3,4,5,6,7,8]
# ---------- u3: Hi, descending-left ----------
W1[3:60:6, 3] = [8,7,6,5,4,3,2,1,0,1]
W1[63:120:6, 3] = [7,6,5,4,3,2,1,0,1,2]
W1[123:180:6, 3] = [6,5,4,3,2,1,0,1,2,3]
W1[183:240:6, 3] = [5,4,3,2,1,0,1,2,3,4]
W1[243:300:6, 3] = [4,3,2,1,0,1,2,3,4,5]
W1[303:360:6, 3] = [3,2,1,0,1,2,3,4,5,6]
W1[363:420:6, 3] = [2,1,0,1,2,3,4,5,6,7]
W1[423:480:6, 3] = [1,0,1,2,3,4,5,6,7,8]
W1[483:540:6, 3] = [0,1,2,3,4,5,6,7,8,9]
# ---------- u4: Lo, ascending-right ----------
W1[3:60:6, 4] = [9,8,7,6,5,4,3,2,1,0]
W1[63:120:6, 4] = [8,7,6,5,4,3,2,1,0,0]
W1[123:180:6, 4] = [7,8,9,8,7,6,5,4,3,2]
W1[183:240:6, 4] = [6,7,8,9,8,7,6,5,4,3]
W1[243:300:6, 4] = [5,6,7,8,9,8,7,6,5,4]
W1[303:360:6, 4] = [6,7,8,9,8,7,6,5,4,3]
W1[363:420:6, 4] = [7,8,9,8,7,6,5,4,3,2]
W1[423:480:6, 4] = [8,9,8,7,6,5,4,3,2,1]
W1[483:540:6, 4] = [9,8,7,6,5,4,3,2,1,0]
# ---------- u5: Hi, ascending-right ----------
W1[3:60:6, 5] = [9,9,8,7,6,5,4,3,2,1]
W1[63:120:6, 5] = [9,8,7,6,5,4,3,2,1,0]
W1[123:180:6, 5] = [8,9,9,9,8,7,6,5,4,3]
W1[183:240:6, 5] = [7,8,9,9,9,8,7,6,5,4]
W1[243:300:6, 5] = [6,7,8,9,9,9,8,7,6,5]
W1[303:360:6, 5] = [7,8,9,9,9,8,7,6,5,4]
W1[363:420:6, 5] = [8,9,9,9,8,7,6,5,4,3]
W1[423:480:6, 5] = [9,9,9,8,7,6,5,4,3,2]
W1[483:540:6, 5] = [9,9,8,7,6,5,4,3,2,1]
# ---------- u6: Lo, ascending-left ----------
W1[3:60:6, 6] = [8,7,6,5,4,3,2,1,0,0]
W1[63:120:6, 6] = [9,8,7,6,5,4,3,2,1,0]
W1[123:180:6, 6] = [6,5,4,3,2,1,0,0,0,1]
W1[183:240:6, 6] = [5,4,3,2,1,0,0,0,1,2]
W1[243:300:6, 6] = [4,3,2,1,0,0,0,1,2,3]
W1[303:360:6, 6] = [5,4,3,2,1,0,0,0,1,2]
W1[363:420:6, 6] = [6,5,4,3,2,1,0,0,0,1]
W1[423:480:6, 6] = [7,6,5,4,3,2,1,0,0,0]
W1[483:540:6, 6] = [8,7,6,5,4,3,2,1,0,0]
# ---------- u7: Hi, ascending-left ----------
W1[3:60:6, 7] = [9,8,7,6,5,4,3,2,1,0]
W1[63:120:6, 7] = [9,9,8,7,6,5,4,3,2,1]
W1[123:180:6, 7] = [7,6,5,4,3,2,1,0,1,2]
W1[183:240:6, 7] = [6,5,4,3,2,1,0,1,2,3]
W1[243:300:6, 7] = [5,4,3,2,1,0,1,2,3,4]
W1[303:360:6, 7] = [6,5,4,3,2,1,0,1,2,3]
W1[363:420:6, 7] = [7,6,5,4,3,2,1,0,1,2]
W1[423:480:6, 7] = [8,7,6,5,4,3,2,1,0,1]
W1[483:540:6, 7] = [9,8,7,6,5,4,3,2,1,0]
# ---------- paddle column (row 9, channel 0): -col for u0..u7 ----------
W1[540:600:6, 0] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 1] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 4] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 5] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 6] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 7] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
# scale: R-units 6*(Lo - p) - 3 ; L-units 6*(p - Hi) - 3
W1[:, 0] *= 6
W1[:, 1] *= -6
W1[:, 2] *= 6
W1[:, 3] *= -6
W1[:, 4] *= 6
W1[:, 5] *= -6
W1[:, 6] *= 6
W1[:, 7] *= -6
b1[0:8] = -3
# ---------- u8: descending gate = 4*(2*ballrow - 2*trailrow + 1 - 2*[trail row 8]) ----------
W1[63:120:6, 8] = 8
W1[123:180:6, 8] = 16
W1[183:240:6, 8] = 24
W1[243:300:6, 8] = 32
W1[303:360:6, 8] = 40
W1[363:420:6, 8] = 48
W1[423:480:6, 8] = 56
W1[483:540:6, 8] = 64
W1[64:120:6, 8] = -8
W1[124:180:6, 8] = -16
W1[184:240:6, 8] = -24
W1[244:300:6, 8] = -32
W1[304:360:6, 8] = -40
W1[364:420:6, 8] = -48
W1[424:480:6, 8] = -56
W1[484:540:6, 8] = -72
b1[8] = 4
# ---------- u9: moving-right gate = 8*(f(ballcol) - trailcol - 0.5), f(0)=1 ----------
W1[3:600:60, 9] = 8
W1[9:600:60, 9] = 8
W1[15:600:60, 9] = 16
W1[21:600:60, 9] = 24
W1[27:600:60, 9] = 32
W1[33:600:60, 9] = 40
W1[39:600:60, 9] = 48
W1[45:600:60, 9] = 56
W1[51:600:60, 9] = 64
W1[57:600:60, 9] = 72
W1[10:600:60, 9] = -8
W1[16:600:60, 9] = -16
W1[22:600:60, 9] = -24
W1[28:600:60, 9] = -32
W1[34:600:60, 9] = -40
W1[40:600:60, 9] = -48
W1[46:600:60, 9] = -56
W1[52:600:60, 9] = -64
W1[58:600:60, 9] = -72
b1[9] = -4
# ---------- Layer 2: AND(comparison, descending gate, right gate) ----------
W2[0, 0] = 5
W2[1, 1] = 5
W2[2, 2] = 5
W2[3, 3] = 5
W2[4, 4] = 5
W2[5, 5] = 5
W2[6, 6] = 5
W2[7, 7] = 5
W2[8, 0:8] = [5, 5, 5, 5, -5, -5, -5, -5]
W2[9, 0:8] = [5, 5, -5, -5, 5, 5, -5, -5]
b2[0:8] = -10
# ---------- Output: right = OR(v0,v2,v4,v6), left = OR(v1,v3,v5,v7), no-op constant ----------
W3[0:8:2, 3] = 20
W3[1:8:2, 1] = 20
b3[3] = 80
b3[1] = 80
b3[0] = 20
```