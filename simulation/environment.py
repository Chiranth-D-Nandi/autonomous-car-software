import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

#codes to be returned to make an action
class Action:
    STOP = 0
    FORWARD = 1
    FORWARD_SLOW = 2
    STEER_LEFT = 3
    STEER_RIGHT = 4

ACTION_NAMES = ["STOP", "FORWARD", "FORWARD_SLOW", "STEER_LEFT", "STEER_RIGHT"]

class Obstacle:
    def __init__(self, x, y, w, h, speed=0, label="obstacle"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.speed = speed
        self.label = label
        self.active = True

    @property
    def rect(self):
        return pygame.Rect(self.x - self.w // 2, self.y - self.h // 2, self.w, self.h)

    def update(self):
        #ui like classic runner game
        #positive speed would move obstacle downwards
        self.y += self.speed

#relative velocity physics
class ObstacleTracker:
    MAX_SPEED = 4.0
    SENSOR_RANGE = 300.0

    def __init__(self, history_size=5):
        self.history = []
        self.speed_history = []
        self.history_size = history_size

    def reset(self):
        self.history = []
        self.speed_history = []

    def update(self, us_center, car_speed_normalized):
        self.history.append(us_center)
        self.speed_history.append(car_speed_normalized)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            self.speed_history.pop(0)

    def analyze(self):
        if len(self.history) < 3: return 0.0, 1.0, 0.0, 1.0, 0.0
        current = self.history[-1]
        if current > 0.9: return 0.0, 1.0, 0.0, 1.0, 0.0

        avg_closing_speed = -np.mean(np.diff(self.history))
        avg_car_speed = np.mean(self.speed_history[:-1])
        car_speed_in_us_units = avg_car_speed * (self.MAX_SPEED / self.SENSOR_RANGE)

        v_object = avg_closing_speed - car_speed_in_us_units
        
        # Static if relative velocity is near zero
        if abs(v_object) < 0.005:
            return 1.0, current, 0.0, 1.0, 0.0
        # Oncoming if it's moving toward us
        elif v_object > 0:
            oncoming_speed = min(v_object * 10.0, 1.0)
            return 0.0, 1.0, 1.0, current, oncoming_speed
        
        return 0.0, 1.0, 0.0, 1.0, 0.0

class DrivingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    WINDOW_W, WINDOW_H = 600, 800
    NUM_LANES, LANE_W = 3, 120
    ROAD_LEFT = (WINDOW_W - NUM_LANES * LANE_W) // 2
    ROAD_RIGHT = ROAD_LEFT + NUM_LANES * LANE_W
    CAR_W, CAR_H = 40, 60
    MAX_SPEED, ACCEL, STEER_SPEED = 4.0, 0.3, 3.0
    SENSOR_RANGE, SENSOR_ANGLES = 300, [-35, 0, 35]

    def __init__(self, render_mode=None, domain_rand=True):
        super().__init__()
        self.render_mode = render_mode
        self.domain_rand = domain_rand
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        self.screen = None
        self.clock = None
        self.tracker = ObstacleTracker()
        self.reset_state_variables()

    def reset_state_variables(self):
        self.car_x, self.car_y, self.speed = 0.0, 0.0, 0.0
        self.obstacles, self.steps, self.score = [], 0, 0.0
        self.passed_obstacle, self.collision = False, False

    def lane_center(self, lane):
        return self.ROAD_LEFT + lane * self.LANE_W + self.LANE_W // 2

    def current_lane(self):
        for i in range(self.NUM_LANES):
            lx = self.ROAD_LEFT + i * self.LANE_W
            if lx <= self.car_x <= lx + self.LANE_W:
                return i
        return -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.reset_state_variables()
        self.car_x = self.lane_center(1)
        self.car_y = self.WINDOW_H - 100
        self.speed = 2.0
        self.tracker.reset()
        self._spawn_scenario()
        return self._get_obs(self._ultrasonic_readings()), {}

    def _spawn_scenario(self):
        truck_y = self.WINDOW_H * (np.random.uniform(0.2, 0.4) if self.domain_rand else 0.3)
        self.obstacles.append(Obstacle(self.lane_center(1), truck_y, 45, 70, 0, "broken_truck"))
        self.obstacles.append(Obstacle(self.lane_center(2), truck_y, 50, 80, 0, "barrier"))
        oncoming_speed = np.random.uniform(3.0, 6.0) if self.domain_rand else 5.0
        oncoming_y = np.random.uniform(-200, 50) if self.domain_rand else -100
        self.obstacles.append(Obstacle(self.lane_center(0), oncoming_y, 40, 65, oncoming_speed, "oncoming_truck"))

    def _cast_ray(self, angle_deg):
        rad = math.radians(angle_deg)
        ox, oy = self.car_x, self.car_y - self.CAR_H // 2
        for d in range(1, self.SENSOR_RANGE + 1, 2):
            px, py = ox + d * math.sin(rad), oy - d * math.cos(rad)
            if not (self.ROAD_LEFT < px < self.ROAD_RIGHT): return d / self.SENSOR_RANGE
            for obs in self.obstacles:
                if obs.active and obs.rect.collidepoint(int(px), int(py)):
                    return d / self.SENSOR_RANGE
        return 1.0

    def _ultrasonic_readings(self):
        return np.array([self._cast_ray(a) for a in self.SENSOR_ANGLES], dtype=np.float32)

    #precomputed sensor readings
    def _get_obs(self, us_readings):
        self.tracker.update(us_readings[1], self.speed / self.MAX_SPEED)
        s_det, s_dist, o_det, o_dist, o_speed = self.tracker.analyze()
        traffic_light, stop_sign = 1.0, 0.0 # Simulates camera: always green, no stop sign
        
        return np.array([
            us_readings[0], us_readings[1], us_readings[2],
            self.speed / self.MAX_SPEED,
            us_readings[0] / (us_readings[0] + us_readings[2] + 1e-6), # Lateral position
            s_det, s_dist, o_det, o_dist, o_speed,
            float(us_readings[0] > 0.3), float(us_readings[2] > 0.3), # Left/Right free
            traffic_light, float(self.passed_obstacle), stop_sign
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        prev_y = self.car_y
        if action == Action.STOP: self.speed = max(self.speed - self.ACCEL * 2, 0)
        elif action == Action.FORWARD: self.speed = min(self.speed + self.ACCEL, self.MAX_SPEED)
        elif action == Action.FORWARD_SLOW:
            target = self.MAX_SPEED * 0.4
            self.speed = max(self.speed - self.ACCEL, target) if self.speed > target else min(self.speed + self.ACCEL * 0.5, target)
        elif action in [Action.STEER_LEFT, Action.STEER_RIGHT]:
            steer_dir = -1 if action == Action.STEER_LEFT else 1
            self.car_x += steer_dir * self.STEER_SPEED
            self.speed = min(self.speed, self.MAX_SPEED * 0.7)
        
        self.car_y -= self.speed
        self.car_x = np.clip(self.car_x, self.ROAD_LEFT + self.CAR_W // 2, self.ROAD_RIGHT - self.CAR_W // 2)
        for obs in self.obstacles:
            obs.update()
            if obs.label == "oncoming_truck" and obs.y > self.WINDOW_H + 100: obs.active = False
    
        car_rect = pygame.Rect(self.car_x-self.CAR_W//2, self.car_y-self.CAR_H//2, self.CAR_W, self.CAR_H)
        self.collision = any(obs.active and obs.rect.colliderect(car_rect) for obs in self.obstacles)
        off_road = not (self.ROAD_LEFT < self.car_x < self.ROAD_RIGHT)
        
        broken_truck = next((o for o in self.obstacles if o.label == "broken_truck"), None)
        just_passed = broken_truck and self.car_y < broken_truck.y - 80 and not self.passed_obstacle
        if just_passed: self.passed_obstacle = True

        reward = 0.0
        us = self._ultrasonic_readings()
        obs = self._get_obs(us)
        
        # Incentives
        reward += (prev_y - self.car_y) * 1.0  # Forward progress
        if just_passed: reward += 50.0       # One-time pass bonus
        if self.car_y < -50 and not self.collision: reward += 200.0 # Goal bonus
        
        # Penalties
        if self.collision: reward -= 200.0
        if off_road: reward -= 150.0
        if min(us) < 0.15: reward -= 3.0
        elif min(us) < 0.3: reward -= 1.0
        reward -= 0.05  # Time penalty
        
        if action in [Action.STEER_LEFT, Action.STEER_RIGHT]: reward -= 0.05 # Gentle steering penalty
        if self.passed_obstacle: # Penalize not returning to center lane
            center_dist = abs(self.car_x - self.lane_center(1))
            reward -= (center_dist / self.LANE_W) * 2.0
        if action == Action.STEER_LEFT and obs[7] > 0.5 and obs[8] < 0.4: reward -= 20.0 # Dangerous merge

        self.score += reward
        terminated = self.collision or off_road
        truncated = self.steps > 800 or self.car_y < -50

        return obs, reward, terminated, truncated, {
            "collision": self.collision, "passed_obstacle": self.passed_obstacle,
            "score": self.score, "steps": self.steps
        }

    def render(self):
        if self.render_mode is None: return None
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Autonomous Driving Sim")
            self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H)) if self.render_mode == "human" else pygame.Surface((self.WINDOW_W, self.WINDOW_H))
            self.clock = pygame.time.Clock()


        self.screen.fill((50, 50, 50))
        pygame.draw.rect(self.screen, (80, 80, 80), (self.ROAD_LEFT, 0, self.NUM_LANES * self.LANE_W, self.WINDOW_H))
        for i in range(1, self.NUM_LANES):
            lx = self.ROAD_LEFT + i * self.LANE_W
            for dy in range(0, self.WINDOW_H, 40): pygame.draw.line(self.screen, (200,200,200), (lx, dy), (lx, dy + 20), 2)
        pygame.draw.line(self.screen, (255,255,255), (self.ROAD_LEFT,0), (self.ROAD_LEFT,self.WINDOW_H), 3)
        pygame.draw.line(self.screen, (255,255,255), (self.ROAD_RIGHT,0), (self.ROAD_RIGHT,self.WINDOW_H), 3)
        for obs in self.obstacles:
            if obs.active: pygame.draw.rect(self.screen, (255,165,0) if obs.speed==0 else (255,0,0), obs.rect)
        pygame.draw.rect(self.screen, (0, 150, 255), (self.car_x-self.CAR_W//2, self.car_y-self.CAR_H//2, self.CAR_W, self.CAR_H))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None