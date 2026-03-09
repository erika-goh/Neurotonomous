import pygame
import math
import numpy as np

class EnhancedGraphics:
    """Enhanced graphics module for Neurotonomous - makes everything look nicer"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Enhanced color palette
        self.colors = {
            'background': (15, 15, 25),
            'car_body': (45, 140, 255),
            'car_accent': (255, 200, 50),
            'car_glow': (100, 180, 255),
            'obstacle': (220, 220, 230),
            'obstacle_border': (180, 180, 200),
            'sensor_active': (255, 100, 100),
            'sensor_inactive': (100, 255, 150),
            'trail': (80, 150, 255),
            'text_primary': (200, 255, 200),
            'text_secondary': (150, 200, 255),
            'grid': (30, 30, 40)
        }
        
        # Trail system for car movement
        self.trail_points = []
        self.max_trail_length = 50
        
        # Particle effects
        self.particles = []
        
    def draw_background_grid(self, screen):
        """Draw a subtle grid background"""
        grid_size = 50
        for x in range(0, self.width, grid_size):
            pygame.draw.line(screen, self.colors['grid'], (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_size):
            pygame.draw.line(screen, self.colors['grid'], (0, y), (self.width, y), 1)
    
    def draw_enhanced_car(self, screen, car):
        """Draw a sleek, futuristic car with glow effects"""
        x, y, angle = int(car.x), int(car.y), car.angle
        
        # Add to trail
        self.trail_points.append((x, y))
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        # Draw trail with fading effect
        for i, (tx, ty) in enumerate(self.trail_points):
            alpha = int(255 * (i / len(self.trail_points)) * 0.3)
            radius = max(1, int(3 * (i / len(self.trail_points))))
            trail_color = (*self.colors['trail'], alpha)
            if i > 0:
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.circle(s, trail_color, (tx, ty), radius)
                screen.blit(s, (0, 0))
        
        # Glow effect
        glow_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for radius in range(25, 15, -2):
            alpha = int(30 * ((25 - radius) / 10))
            glow_color = (*self.colors['car_glow'], alpha)
            pygame.draw.circle(glow_surface, glow_color, (x, y), radius)
        screen.blit(glow_surface, (0, 0))
        
        # Car body - draw as a pointed triangle/arrow shape
        rad = math.radians(angle)
        car_length = 18
        car_width = 12
        
        # Calculate car vertices
        front = (x + car_length * math.cos(rad), y + car_length * math.sin(rad))
        back_left = (
            x + car_width * math.cos(rad + math.pi * 0.6),
            y + car_width * math.sin(rad + math.pi * 0.6)
        )
        back_right = (
            x + car_width * math.cos(rad - math.pi * 0.6),
            y + car_width * math.sin(rad - math.pi * 0.6)
        )
        
        # Draw car shadow
        shadow_offset = 3
        shadow_points = [
            (front[0] + shadow_offset, front[1] + shadow_offset),
            (back_left[0] + shadow_offset, back_left[1] + shadow_offset),
            (back_right[0] + shadow_offset, back_right[1] + shadow_offset)
        ]
        pygame.draw.polygon(screen, (0, 0, 0, 100), shadow_points)
        
        # Draw main car body
        car_points = [front, back_left, back_right]
        pygame.draw.polygon(screen, self.colors['car_body'], car_points)
        pygame.draw.polygon(screen, self.colors['car_accent'], car_points, 2)
        
        # Draw direction indicator (small circle at front)
        pygame.draw.circle(screen, self.colors['car_accent'], 
                         (int(front[0]), int(front[1])), 4)
        pygame.draw.circle(screen, self.colors['car_body'], 
                         (int(front[0]), int(front[1])), 2)
        
        # Speed indicator (small lines behind car when moving)
        if car.speed > 1:
            for i in range(3):
                speed_length = car.speed * 2
                offset = -car_length - i * 5
                sx = x + offset * math.cos(rad)
                sy = y + offset * math.sin(rad)
                ex = sx - speed_length * math.cos(rad)
                ey = sy - speed_length * math.sin(rad)
                alpha = int(100 - i * 30)
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.line(s, (*self.colors['trail'], alpha), 
                               (sx, sy), (ex, ey), 2)
                screen.blit(s, (0, 0))
    
    def draw_enhanced_sensors(self, screen, car, state, sensor_angles, sensor_length):
        """Draw sensors with gradient effect and distance indicators"""
        for i, (offset, dist_normalized) in enumerate(zip(sensor_angles, state)):
            rad = math.radians(car.angle + offset)
            
            # Calculate sensor end point
            actual_dist = dist_normalized * sensor_length
            sx = int(car.x + actual_dist * math.cos(rad))
            sy = int(car.y + actual_dist * math.sin(rad))
            
            # Color based on distance (red when close, green when far)
            if dist_normalized < 0.3:
                color = self.colors['sensor_active']
                alpha = 200
            else:
                color = self.colors['sensor_inactive']
                alpha = 100
            
            # Draw sensor line with transparency
            sensor_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(sensor_surface, (*color, alpha), 
                           (car.x, car.y), (sx, sy), 2)
            screen.blit(sensor_surface, (0, 0))
            
            # Draw sensor endpoint glow
            if dist_normalized < 1.0:  # Hit something
                for r in range(8, 3, -1):
                    alpha_glow = int(150 * ((8 - r) / 5))
                    glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*color, alpha_glow), (sx, sy), r)
                    screen.blit(glow_surf, (0, 0))
            
            # Draw small dot at sensor origin
            pygame.draw.circle(screen, color, (int(car.x), int(car.y)), 2)
    
    def draw_enhanced_obstacles(self, screen, obstacles):
        """Draw obstacles with 3D effect and shadows"""
        for obs in obstacles:
            # Shadow
            shadow_rect = obs.copy()
            shadow_rect.x += 4
            shadow_rect.y += 4
            s = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
            s.fill((*self.colors['background'], 150))
            screen.blit(s, (shadow_rect.x, shadow_rect.y))
            
            # Main obstacle with gradient effect
            pygame.draw.rect(screen, self.colors['obstacle'], obs)
            pygame.draw.rect(screen, self.colors['obstacle_border'], obs, 3)
            
            # Highlight on top-left for 3D effect
            highlight_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height // 4)
            highlight_surf = pygame.Surface((obs.width, obs.height // 4), pygame.SRCALPHA)
            highlight_surf.fill((255, 255, 255, 30))
            screen.blit(highlight_surf, (highlight_rect.x, highlight_rect.y))
    
    def draw_enhanced_text(self, screen, text, size, x, y, color=None, shadow=True):
        """Draw text with shadow and glow effect"""
        if color is None:
            color = self.colors['text_primary']
        
        font = pygame.font.SysFont('Arial', size, bold=True)
        
        # Shadow
        if shadow:
            shadow_surface = font.render(text, True, (0, 0, 0))
            shadow_rect = shadow_surface.get_rect(center=(x + 2, y + 2))
            screen.blit(shadow_surface, shadow_rect)
        
        # Main text
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    
    def draw_title_screen(self, screen):
        """Draw an enhanced title screen"""
        screen.fill(self.colors['background'])
        self.draw_background_grid(screen)
        
        # Animated glow effect behind title
        for i in range(5):
            alpha = int(30 - i * 5)
            size_offset = i * 10
            glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.draw_enhanced_text(glow_surf, "NEUROTONOMOUS", 72 + size_offset, 
                                  self.width // 2, self.height // 3, 
                                  (*self.colors['car_glow'], alpha), shadow=False)
            screen.blit(glow_surf, (0, 0))
        
        # Main title
        self.draw_enhanced_text(screen, "NEUROTONOMOUS", 72, 
                              self.width // 2, self.height // 3, 
                              self.colors['car_accent'])
        
        self.draw_enhanced_text(screen, "AI Driving Simulator", 36, 
                              self.width // 2, self.height // 2, 
                              self.colors['text_secondary'])
        
        self.draw_enhanced_text(screen, "Press any key to start training", 28, 
                              self.width // 2, self.height * 2 // 3, 
                              self.colors['text_primary'])
        
        # Draw a demo car on title screen
        demo_car_x = self.width // 2
        demo_car_y = self.height * 3 // 4
        angle = (pygame.time.get_ticks() / 10) % 360
        
        rad = math.radians(angle)
        orbit_radius = 80
        car_x = demo_car_x + orbit_radius * math.cos(math.radians(angle))
        car_y = demo_car_y + orbit_radius * math.sin(math.radians(angle))
        
        # Mock car object for drawing
        class MockCar:
            def __init__(self, x, y, angle):
                self.x, self.y, self.angle = x, y, angle
                self.speed = 3
        
        demo_car = MockCar(car_x, car_y, angle + 90)
        self.draw_enhanced_car(screen, demo_car)
    
    def draw_crash_effect(self, screen, car):
        """Draw explosion effect when car crashes"""
        # Create expanding rings
        for i in range(5):
            radius = 20 + i * 15
            alpha = int(255 - i * 50)
            crash_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.circle(crash_surf, (255, 100, 50, alpha), 
                             (int(car.x), int(car.y)), radius, 3)
            screen.blit(crash_surf, (0, 0))
        
        # Draw "CRASHED!" text
        self.draw_enhanced_text(screen, "CRASHED!", 80, 
                              self.width // 2, self.height // 2, 
                              (255, 50, 50))
    
    def draw_stats_panel(self, screen, episode, steps, epsilon, reward=None):
        """Draw a sleek stats panel"""
        panel_height = 80
        panel_surf = pygame.Surface((self.width, panel_height), pygame.SRCALPHA)
        panel_surf.fill((*self.colors['background'], 200))
        screen.blit(panel_surf, (0, 0))
        
        # Draw stats
        self.draw_enhanced_text(screen, f"Episode: {episode}", 24, 
                              self.width // 6, 25, self.colors['text_secondary'])
        self.draw_enhanced_text(screen, f"Steps: {steps}", 24, 
                              self.width // 2, 25, self.colors['text_secondary'])
        self.draw_enhanced_text(screen, f"Epsilon: {epsilon:.3f}", 24, 
                              self.width * 5 // 6, 25, self.colors['text_secondary'])
        
        if reward is not None:
            self.draw_enhanced_text(screen, f"Reward: {reward:.1f}", 20, 
                                  self.width // 2, 55, self.colors['car_accent'])
    
    def reset_trail(self):
        """Clear the trail points"""
        self.trail_points = []