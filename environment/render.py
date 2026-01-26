import pygame
import numpy as np
from .context import Actions, ACTION_NAMES

def render_frame(env):
    env.window_size = 512  # The size of the grid portion of the PyGame window
    env.panel_width = 600  # Width for the info panel.
    env.window_width = env.window_size + env.panel_width  # Total window width.
    env.window_height = env.window_size  # Window height remains same as grid height.
    env.window = None
    env.clock = None

    pygame.font.init()
    env.font = pygame.font.SysFont("Arial", 20)

    if env.window is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        env.window = pygame.display.set_mode((env.window_width, env.window_height))
    if env.clock is None and env.render_mode == "human":
        env.clock = pygame.time.Clock()

    canvas = pygame.Surface((env.window_width, env.window_height))
    canvas.fill((255, 255, 255))

    # Grid
    pix_square_size = env.window_size / env.size
    for x in range(env.size + 1):
        pygame.draw.line(canvas, (0, 0, 0),
                         (0, pix_square_size * x),
                         (env.window_size, pix_square_size * x), 2)
        pygame.draw.line(canvas, (0, 0, 0),
                         (pix_square_size * x, 0),
                         (pix_square_size * x, env.window_size), 2)

    # Static locations
    colour_map = {"PHC": (200, 200, 200), "ICU": (50, 50, 50), "SocialService": (100, 100, 100)}
    label_map = {"PHC": "PHC", "ICU": "ICU", "SocialService": "SS"}

    for name, info in env.context.locations.items():
        base = np.array(info["pos"])
        w, h = info.get("size", (1, 1))
        col = colour_map.get(name, (150, 150, 150))
        label = label_map.get(name, name[:2].upper())

        for dx in range(w):
            for dy in range(h):
                rect = pygame.Rect((base + (dx, dy)) * pix_square_size, (pix_square_size, pix_square_size))
                pygame.draw.rect(canvas, col, rect)
                if dx == 0 and dy == 0:
                    txt_surf = env.font.render(label, True, (0, 0, 0))
                    canvas.blit(txt_surf, (rect.x + 3, rect.y + 3))

    # PEH agents
    for agent in env.peh_agents:
        color = env._get_health_color(agent)
        center = (agent.location + 0.5) * pix_square_size
        pygame.draw.circle(canvas, color, center, pix_square_size / 3)

    # Soc service agents
    for soc_agent in env.socserv_agents:
        center = (soc_agent.location + 0.5) * pix_square_size
        pygame.draw.circle(canvas, (128, 128, 128), center, pix_square_size / 5)

    # Info panel
    text_color = (0, 0, 0)
    panel_x = env.window_size + 10
    panel_x2 = env.window_size + 250

    canvas.blit(env.font.render(f"Steps: {env.step_count}", True, text_color), (panel_x, 10))
    #canvas.blit(env.font.render(f"Total Reward: {env.cumulative_reward:.1f}", True, text_color), (panel_x, 35))
    canvas.blit(env.font.render(f"Budget: {env.context.healthcare_budget}", True, text_color), (panel_x, 60))

    for i, agent in enumerate(env.agents):
        a = env.peh_agents[i]
        y_base = 100 + i * 230
        canvas.blit(env.font.render(f"{agent} — Health: {a.health_state:.2f}", True, text_color), (panel_x, y_base))
        canvas.blit(env.font.render(f"Status: {a.administrative_state}", True, text_color), (panel_x, y_base + 20))
        #canvas.blit(env.font.render(f"Steps: {env.agent_step_counts[agent]}", True, text_color), (panel_x, y_base + 40))
        canvas.blit(env.font.render(f"Reward: {env.rewards[agent]:.2f}", True, text_color), (panel_x, y_base + 60))

        # Possible actions
        canvas.blit(env.font.render("Possible Actions:", True, text_color), (panel_x, y_base + 90))
        for j, action in enumerate(env.possible_actions_history[agent][-5:]):
            name = ACTION_NAMES.get(action, str(action))
            canvas.blit(env.font.render(name, True, text_color), (panel_x, y_base + 110 + j * 20))

        # Impossible actions
        canvas.blit(env.font.render("Impossible Actions:", True, text_color), (panel_x, y_base + 100))
        for j, action in enumerate(env.impossible_actions_history[agent][-5:]):
            name = ACTION_NAMES.get(action, str(action))
            canvas.blit(env.font.render(name, True, text_color), (panel_x, y_base + 240 + j * 20))

        # Capabilities
        canvas.blit(env.font.render("Capabilities:", True, text_color), (panel_x2, y_base))
        for j, (label, status) in enumerate(env.capabilities[agent].items()):
            color = (0, 150, 0) if status else (200, 0, 0)
            canvas.blit(env.font.render(label, True, color), (panel_x2, y_base + 20 + j * 20))

        # Functionings
        canvas.blit(env.font.render("Functionings:", True, text_color), (panel_x2, y_base + 110))
        for j, (label, status) in enumerate(env.functionings[agent].items()):
            color = (0, 150, 0) if status else (200, 0, 0)
            canvas.blit(env.font.render(label, True, color), (panel_x2, y_base + 130 + j * 20))

    if env.render_mode == "human":
        env.window.blit(canvas, canvas.get_rect())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.display.update()
        env.clock.tick(env.metadata["render_fps"])
    
    else:
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
