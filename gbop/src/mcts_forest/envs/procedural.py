import numpy as np
from numba import njit
import math

@njit(cache=True)
def river_swim_step(s, a, nr, nl, tl):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    sid = s // (tl + 1)
    
    reward = 0.0
    all_at_goal = True
    all_swim_up = True
    goal = nl - 1
    
    next_sid = 0
    factor = 1
    temp_sid = sid
    for i in range(nr):
        pos = temp_sid % nl
        temp_sid //= nl
        act = (a >> i) & 1
        
        if pos == 0 and act == 0: reward += 0.1
        if pos == goal and act == 1: reward += 1.0
        
        if pos != goal: all_at_goal = False
        if act != 1: all_swim_up = False
        
        next_pos = pos
        if act == 0:
            next_pos = max(0, pos - 1)
        else:
            p = np.random.random()
            if pos == 0:
                next_pos = 0 if p < 0.4 else 1
            elif pos == goal:
                next_pos = goal - 1 if p < 0.4 else goal
            else:
                if p < 0.05: next_pos = pos - 1
                elif p < 0.65: next_pos = pos
                else: next_pos = pos + 1
        
        next_sid += next_pos * factor
        factor *= nl
        
    if all_at_goal and all_swim_up:
        reward += float(nr)
    
    reward /= (2.0 * nr)
    next_t = t + 1
    next_s = next_sid * (tl + 1) + next_t
    return next_s, reward, (next_t >= tl)

@njit(cache=True)
def river_swim_rollout(s, nr, nl, tl, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 1 << nr)
        curr_s, r, done = river_swim_step(curr_s, a, nr, nl, tl)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

@njit(cache=True)
def four_rooms_step(s, a, n, gs, tl, slip, doors, gx, gy):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    pos = s // (tl + 1)
    x, y = pos % gs, pos // gs
    
    eff_a = a
    if slip:
        p = np.random.random()
        if p < 0.25: eff_a = (a + 3) % 4
        elif p >= 0.75: eff_a = (a + 1) % 4
    
    dx, dy = 0, 0
    if eff_a == 0: dx = -1
    elif eff_a == 1: dy = 1
    elif eff_a == 2: dx = 1
    elif eff_a == 3: dy = -1
    
    nx, ny = x + dx, y + dy
    
    blocked = False
    if nx < 0 or nx >= gs or ny < 0 or ny >= gs:
        blocked = True
    else:
        is_on_wall_line = (nx == n or ny == n)
        if is_on_wall_line:
            idx = ny * gs + nx
            is_door = False
            for d in doors:
                if idx == d:
                    is_door = True
                    break
            if not is_door:
                blocked = True
    
    if blocked:
        nx, ny = x, y
        
    next_t = t + 1
    reward = 0.0
    done = False
    if nx == gx and ny == gy:
        reward = 1.0 - 0.9 * (float(next_t) / float(tl))
        done = True
    elif next_t >= tl:
        done = True
        
    next_s = (ny * gs + nx) * (tl + 1) + next_t
    return next_s, reward, done

@njit(cache=True)
def four_rooms_rollout(s, n, gs, tl, slip, doors, gx, gy, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = four_rooms_step(curr_s, a, n, gs, tl, slip, doors, gx, gy)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

@njit(cache=True)
def passenger_grid_step(s, a, w, h, tl, slip, pass_pos, gx, gy):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    encoded = s // (tl + 1)
    mask = encoded & 7
    spatial_id = encoded >> 3
    x, y = spatial_id % w, spatial_id // w
    
    eff_a = a
    if slip:
        p = np.random.random()
        if p < 0.25: eff_a = (a + 3) % 4
        elif p >= 0.75: eff_a = (a + 1) % 4
        
    dx, dy = 0, 0
    if eff_a == 0: dx = -1
    elif eff_a == 1: dy = 1
    elif eff_a == 2: dx = 1
    elif eff_a == 3: dy = -1
    
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        nx, ny = x, y
        
    n_mask = mask
    for i in range(3):
        if nx == pass_pos[i, 0] and ny == pass_pos[i, 1]:
            n_mask |= (1 << i)
            
    next_t = t + 1
    reward = 0.0
    done = False
    if nx == gx and ny == gy:
        picked = 0
        for i in range(3):
            if (n_mask >> i) & 1: picked += 1
        rewards_arr = np.array([0.0, 1.0, 3.0, 7.0])
        reward = rewards_arr[picked]
        done = True
    elif next_t >= tl:
        done = True
        
    next_sid = (ny * w + nx) << 3 | n_mask
    next_s = next_sid * (tl + 1) + next_t
    return next_s, reward, done

@njit(cache=True)
def passenger_grid_rollout(s, w, h, tl, slip, pass_pos, gx, gy, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = passenger_grid_step(curr_s, a, w, h, tl, slip, pass_pos, gx, gy)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

@njit(cache=True)
def sysadmin_ring_step(s, a, nc, tl, probs):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    mask = s // (tl + 1)
    
    next_mask = 0
    for i in range(nc):
        if a == i:
            next_mask |= (1 << i)
        else:
            prev_machine = (i - 1 + nc) % nc
            prev_running = (mask >> prev_machine) & 1
            self_running = (mask >> i) & 1
            p = probs[(self_running << 1) | prev_running]
            if np.random.random() < p:
                next_mask |= (1 << i)
                
    count = 0
    for i in range(nc):
        if (next_mask >> i) & 1: count += 1
    reward = float(count) / float(nc)
    
    next_t = t + 1
    next_s = next_mask * (tl + 1) + next_t
    return next_s, reward, (next_t >= tl)

@njit(cache=True)
def sysadmin_ring_rollout(s, nc, tl, probs, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, nc + 1)
        curr_s, r, done = sysadmin_ring_step(curr_s, a, nc, tl, probs)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

@njit(cache=True)
def frozen_lake_step(s, a, grid, w, h, slip):
    x, y = s % w, s // w
    
    if slip:
        p = np.random.random()
        if p < 1.0/3.0: eff_a = (a - 1) % 4
        elif p < 2.0/3.0: eff_a = a
        else: eff_a = (a + 1) % 4
    else:
        eff_a = a
        
    dx, dy = 0, 0
    if eff_a == 0: dx = -1
    elif eff_a == 1: dy = 1
    elif eff_a == 2: dx = 1
    elif eff_a == 3: dy = -1
    
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        nx, ny = x, y
        
    next_s = ny * w + nx
    cell = grid[next_s]
    
    reward = 1.0 if cell == 2 else 0.0
    done = (cell == 1 or cell == 2)
    return next_s, reward, done

@njit(cache=True)
def frozen_lake_rollout(s, grid, w, h, slip, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = frozen_lake_step(curr_s, a, grid, w, h, slip)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

