# generate_evolution_gif.py
import os
os.environ['MPLBACKEND'] = 'Agg'  # ← Força backend sense display

import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import json
from environment.model import GridMAInequityEnv
from environment.context import Context
from learning.utils import get_state, action_mask_from_classify, masked_argmax
from learning.qpbrs import plot_policy_summary_comparison
import io
from PIL import Image

def load_q_tables_and_profiles(run_dir):
    """Load Q-tables and profiles from a run directory."""
    models_dir = os.path.join(run_dir, "models")
    
    q_tables_on = np.load(os.path.join(models_dir, "q_tables_advice_ON.npy"), allow_pickle=True).item()
    q_tables_off = np.load(os.path.join(models_dir, "q_tables_advice_OFF.npy"), allow_pickle=True).item()
    
    with open("output/peh_sample8.json", "r") as f:
        profiles = json.load(f)
    
    return q_tables_on, q_tables_off, profiles

def run_simulation_and_track(env, q_tables, max_steps=100):
    """
    Run simulation and track all histories needed for plot_policy_summary_comparison.
    """
    
    # Initialize tracking dictionaries
    bh_trace = {ag: [] for ag in env.possible_agents}
    af_trace = {ag: [] for ag in env.possible_agents}
    health_trace = {ag: [] for ag in env.possible_agents}
    admin_trace = {ag: [] for ag in env.possible_agents}
    
    init_admin = {}
    init_trust = {}
    
    # Store initial state
    for ag in env.possible_agents:
        idx = env.agent_name_mapping[ag]
        peh = env.peh_agents[idx]
        init_admin[ag] = peh.administrative_state
        init_trust[ag] = peh.trust_type
        
        # Initial values
        health_trace[ag].append(peh.health_state)
        admin_trace[ag].append(1 if peh.administrative_state == "registered" else 0)
        bh_trace[ag].append(0.0)
        af_trace[ag].append(0.0)
    
    # Get initial budgets
    ctx = env.context
    init_health_budget = getattr(ctx, "healthcare_budget", 10000.0)
    init_social_budget = getattr(ctx, "social_service_budget", 5000.0)
    
    snapshots = []
    step_count = 0
    snapshot_interval = 2
    
    # Initial snapshot
    snapshots.append({
        'step': 0,
        'bh_trace': {k: list(v) for k, v in bh_trace.items()},
        'af_trace': {k: list(v) for k, v in af_trace.items()},
        'health_trace': {k: list(v) for k, v in health_trace.items()},
        'admin_trace': {k: list(v) for k, v in admin_trace.items()},
    })
    
    # Run simulation
    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        
        if agent not in q_tables:
            action = env.action_space(agent).sample()
        else:
            idx = env.agent_name_mapping[agent]
            peh_agent = env.peh_agents[idx]
            obs = env.observe(agent)
            state = get_state(obs, peh_agent)
            mask = action_mask_from_classify(env, agent)
            action = masked_argmax(q_tables[agent][state], mask)
        
        env.step(action)
        step_count += 1
        
        # Update all traces
        for ag in env.possible_agents:
            idx = env.agent_name_mapping[ag]
            peh = env.peh_agents[idx]
            
            health_trace[ag].append(peh.health_state)
            admin_trace[ag].append(1 if peh.administrative_state == "registered" else 0)
            
            if hasattr(env, 'capabilities') and ag in env.capabilities:
                caps = env.capabilities[ag]
                bh_trace[ag].append(1.0 - float(caps.get("Being able to have good health", 0)))
                af_trace[ag].append(1.0 - float(caps.get("Being able to have adequate shelter", 0)))
            else:
                bh_trace[ag].append(1.0 if peh.health_state >= 3.0 else 0.0)
                af_trace[ag].append(1.0 if peh.administrative_state == "registered" else 0.5)
        
        # Take snapshot at intervals
        if step_count % snapshot_interval == 0 or not env.agents:
            snapshots.append({
                'step': step_count,
                'bh_trace': {k: list(v) for k, v in bh_trace.items()},
                'af_trace': {k: list(v) for k, v in af_trace.items()},
                'health_trace': {k: list(v) for k, v in health_trace.items()},
                'admin_trace': {k: list(v) for k, v in admin_trace.items()},
            })
    
    return snapshots, init_admin, init_trust, init_health_budget, init_social_budget


def fig_to_array(fig):
    """Convert matplotlib figure to numpy array (RGB only)."""
    # Force a draw to ensure everything is rendered
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()
    # Convert to numpy array
    img_array = np.asarray(buf)
    
    # Convert RGBA to RGB (remove alpha channel)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Ensure uint8
    img_array = img_array.astype(np.uint8)
    
    return img_array


def create_comparison_evolution_gif(run_dir, output_name="policy_evolution_comparison.gif", 
                                   output_format="mp4", max_steps=100, debug_frames=False):
    """
    Generate evolution video/GIF using your existing plot_policy_summary_comparison.
    
    Args:
        output_format: 'gif' or 'mp4'
        debug_frames: If True, save individual frames as PNG for debugging
    """
    
    q_tables_on, q_tables_off, profiles = load_q_tables_and_profiles(run_dir)
    
    size = 7
    num_peh = len(profiles)
    num_sw = 15
    
    # Create environments
    print("Setting up environments...")
    ctx_on = Context(grid_size=size)
    ctx_on.set_scenario(policy_inclusive_healthcare=True)
    env_on = GridMAInequityEnv(
        context=ctx_on, render_mode="rgb_array", size=size,
        num_peh=num_peh, num_social_agents=num_sw,
        peh_profiles=profiles, max_steps=150
    )
    env_on.reset(options={"peh_profiles": profiles})
    
    ctx_off = Context(grid_size=size)
    ctx_off.set_scenario(policy_inclusive_healthcare=False)
    env_off = GridMAInequityEnv(
        context=ctx_off, render_mode="rgb_array", size=size,
        num_peh=num_peh, num_social_agents=num_sw,
        peh_profiles=profiles, max_steps=150
    )
    env_off.reset(options={"peh_profiles": profiles})
    
    # Run both simulations
    print("Running POLICY ON simulation...")
    snapshots_on, init_admin_on, init_trust_on, init_hb_on, init_sb_on = \
        run_simulation_and_track(env_on, q_tables_on, max_steps=max_steps)
    
    print("Running POLICY OFF simulation...")
    snapshots_off, init_admin_off, init_trust_off, init_hb_off, init_sb_off = \
        run_simulation_and_track(env_off, q_tables_off, max_steps=max_steps)
    
    # Generate frames
    print(f"Generating {len(snapshots_on)} frames...")
    frames = []
    
    if debug_frames:
        debug_dir = os.path.join(run_dir, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
    
    for i, (snap_on, snap_off) in enumerate(zip(snapshots_on, snapshots_off)):
        print(f"  Frame {i+1}/{len(snapshots_on)} (step {snap_on['step']})", end='\r')
        
        # Create figure with your existing function
        plot_policy_summary_comparison(
            env_on=env_on,
            bh_on=snap_on['bh_trace'],
            af_on=snap_on['af_trace'],
            health_on=snap_on['health_trace'],
            admin_on=snap_on['admin_trace'],
            init_admin_on=init_admin_on,
            init_trust_on=init_trust_on,
            init_health_budget_on=init_hb_on,
            init_social_budget_on=init_sb_on,
            
            env_off=env_off,
            bh_off=snap_off['bh_trace'],
            af_off=snap_off['af_trace'],
            health_off=snap_off['health_trace'],
            admin_off=snap_off['admin_trace'],
            init_admin_off=init_admin_off,
            init_trust_off=init_trust_off,
            init_health_budget_off=init_hb_off,
            init_social_budget_off=init_sb_off,
            
            title_on=f"Policy ON - Step {snap_on['step']}",
            title_off=f"Policy OFF - Step {snap_off['step']}",
            show_social_workers=True,
        )
        
        # Get current figure and convert to array
        fig = plt.gcf()
        img_array = fig_to_array(fig)
        
        # Debug: save individual frame
        if debug_frames:
            frame_path = os.path.join(debug_dir, f"frame_{i:03d}.png")
            Image.fromarray(img_array).save(frame_path)
            print(f"  Saved debug frame: {frame_path}")
        
        frames.append(img_array)
        plt.close('all')
    
    print(f"\nSaving as {output_format.upper()}...")
    
    # Save based on format
    if output_format.lower() == 'mp4':
        output_path = os.path.join(run_dir, output_name.replace('.gif', '.mp4'))
        # Use imageio with ffmpeg
        imageio.mimsave(output_path, frames, fps=2, codec='libx264', 
                       quality=8, pixelformat='yuv420p')
    else:  # gif
        output_path = os.path.join(run_dir, output_name)
        # Convert to PIL Images for better GIF handling
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=800,  # ms per frame
            loop=0,
            optimize=False  # Important: don't optimize, it can cause blank frames
        )
    
    print(f"✓ Video saved: {output_path}")
    
    if debug_frames:
        print(f"✓ Debug frames saved in: {debug_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        output_dirs = [os.path.join("output", d) for d in os.listdir("output")
                      if os.path.isdir(os.path.join("output", d)) and d.startswith("run_")]
        if not output_dirs:
            print("❌ No run directories found in output/")
            sys.exit(1)
        run_dir = max(output_dirs, key=os.path.getctime)
    
    print(f"Using run directory: {run_dir}")
    
    # Try MP4 first (usually works better)
    try:
        create_comparison_evolution_gif(
            run_dir, 
            output_name="policy_evolution_comparison.mp4",
            output_format="mp4",
            max_steps=100,
            debug_frames=False  # Set to True to debug
        )
    except Exception as e:
        print(f"⚠️  MP4 failed: {e}")
        print("Trying GIF instead...")
        create_comparison_evolution_gif(
            run_dir,
            output_name="policy_evolution_comparison.gif",
            output_format="gif",
            max_steps=100,
            debug_frames=True  # Save debug frames
        )
    
    print("\n✓ Evolution video generated successfully!")