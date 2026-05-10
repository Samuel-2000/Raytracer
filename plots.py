import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def load_benchmark_data(csv_path='benchmark_results.csv'):
    """Načíta dáta z CSV súboru a zoradí ich od 0 po 1."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Súbor {csv_path} neexistuje. Spustite najprv benchmark.")
    
    df = pd.read_csv(csv_path)
    df.sort_values(by=['bvh', 'dynamic_bvh', 'simd'], ascending=[True, True, True], inplace=True)
    return df

def plot_technique_comparison(df, output_path='benchmark_comparison.png'):
    """Vykreslí porovnanie techník pre každú scénu (stĺpcový graf)."""
    df['combination'] = df.apply(lambda row: 
        f"Build={int(row['bvh'])}, Fit={int(row['dynamic_bvh'])}, SIMD={int(row['simd'])}", axis=1)
    
    scenes = df['scene'].unique()
    n_scenes = len(scenes)
    
    fig, axes = plt.subplots(n_scenes, 1, figsize=(12, 5*n_scenes), squeeze=False)
    fig.suptitle('Porovnanie výkonu optimalizačných techník', fontsize=16)
    
    for idx, scene in enumerate(scenes):
        ax = axes[idx, 0]
        scene_df = df[df['scene'] == scene]
        colors = ['green' if bvh else 'red' for bvh in scene_df['bvh']]
        
        bars = ax.barh(scene_df['combination'], scene_df['time'], color=colors)
        ax.set_title(f'Scéna: {scene}')
        ax.set_xlabel('Čas (s)')
        ax.set_ylabel('Kombinácia techník')
        
        for bar, time_val in zip(bars, scene_df['time']):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{time_val:.2f}s', va='center', fontsize=8)
    
    green_patch = mpatches.Patch(color='green', label='Build=1')
    red_patch = mpatches.Patch(color='red', label='Build=0')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=12)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_simd_comparison(df, output_path='simd_comparison.png'):
    """Vertikálny graf rozdelený na 'Bez SIMD' a 'So SIMD' (bez 'simple' scény)."""
    scenes = [s for s in df['scene'].unique() if s != 'simple']
    
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 6), sharey=False)
    
    if len(scenes) == 1:
        axes = [axes]
        
    x = np.arange(2)
    width = 0.25
    
    combinations = [
        (True, True, 'Build=1, Fit=1', '#2ecc71'),   
        (True, False, 'Build=1, Fit=0', '#f39c12'),  
        (False, False, 'Build=0, Fit=0', '#e74c3c')  
    ]
    
    for i, scene in enumerate(scenes):
        ax = axes[i]
        scene_df = df[df['scene'] == scene]
        
        for j, (bvh, dyn, label, color) in enumerate(combinations):
            vals = []
            for simd in [False, True]:
                val = scene_df[(scene_df['bvh'] == bvh) & (scene_df['dynamic_bvh'] == dyn) & (scene_df['simd'] == simd)]['time'].values
                vals.append(val[0] if len(val) > 0 else np.nan)
            
            pos = x + (j - 1) * width
            ax.bar(pos, vals, width, label=label, color=color, edgecolor='black', alpha=0.9)
            
            for idx_pos, v in zip(pos, vals):
                if not np.isnan(v):
                    ax.text(idx_pos, v + (np.nanmax(vals)*0.02 if not np.isnan(np.nanmax(vals)) else 0.1), 
                            f"{v:.2f}s", ha='center', va='bottom', fontsize=8)

        ax.set_title(f'Scéna: {scene}')
        ax.set_xticks(x)
        ax.set_xticklabels(['Bez SIMD (0)', 'So SIMD (1)'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Čas (s)')
        ax.grid(True, axis='y', alpha=0.3)
            
    plt.suptitle("Vplyv SIMD vektorizácie naprieč technikami", fontsize=16)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=12)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15) 
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_speedup(df, output_path='speedup.png'):
    """Vykreslí zrýchlenie oproti základnej kombinácii."""
    df['combination'] = df.apply(lambda row: 
        f"Build={int(row['bvh'])}, Fit={int(row['dynamic_bvh'])}, SIMD={int(row['simd'])}", axis=1)
    
    all_combinations = list(df['combination'].unique())
    
    color_map = {
        'simple': 'tab:blue',
        'many_objects_100': '#e74c3c',
        'many_objects_1k': 'tab:green',
        'many_objects_10k': '#f1c40f'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for scene in df['scene'].unique():
        scene_df = df[df['scene'] == scene]
        
        base_df = scene_df[(scene_df['bvh'] == False) & 
                           (scene_df['dynamic_bvh'] == False) & 
                           (scene_df['simd'] == False)]
        
        if not base_df.empty:
            base_time = base_df['time'].iloc[0]
        else:
            base_time = scene_df['time'].max()
            
        if pd.isna(base_time):
            continue
            
        x_indices = []
        y_vals = []
        
        for i, comb in enumerate(all_combinations):
            row = scene_df[scene_df['combination'] == comb]
            if not row.empty:
                x_indices.append(i)
                speedup = base_time / row['time'].iloc[0]
                y_vals.append(speedup)
        
        if x_indices:
            c = color_map.get(scene, None)
            ax.plot(x_indices, y_vals, marker='o', label=scene, linewidth=2, color=c)

    ax.set_xticks(range(len(all_combinations)))
    ax.set_xticklabels(all_combinations, rotation=45, ha='right')
    
    ax.set_xlabel('Kombinácia techník')
    ax.set_ylabel('Zrýchlenie (násobok)')
    ax.set_title('Zrýchlenie oproti základnej konfigurácii (alebo najpomalšiemu behu)', pad=35)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(df['scene'].unique()), frameon=False)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_bvh_impact(df, output_path='build_impact.png'):
    """Vykreslí vplyv Build na výkon pre každú scénu."""
    base = df[(df['dynamic_bvh'] == False) & (df['simd'] == False)]
    
    scenes = base['scene'].unique()
    x = np.arange(len(scenes))
    width = 0.35
    
    bvh_on = []
    bvh_off = []
    
    for scene in scenes:
        val_on = base[(base['scene'] == scene) & (base['bvh'] == True)]['time'].values
        val_off = base[(base['scene'] == scene) & (base['bvh'] == False)]['time'].values
        bvh_on.append(val_on[0] if len(val_on) > 0 else np.nan)
        bvh_off.append(val_off[0] if len(val_off) > 0 else np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bvh_on, width, label='Build=1', color='green')
    bars2 = ax.bar(x + width/2, bvh_off, width, label='Build=0', color='red')
    
    ax.set_xlabel('Scéna')
    ax.set_ylabel('Čas (s)')
    ax.set_title('Vplyv Build na rýchlosť renderovania (Fit=0, SIMD=0)', pad=35)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_bvh_aggregated(df, output_path='build_aggregated.png'):
    scenes = df['scene'].unique()
    x = np.arange(len(scenes))
    width = 0.35
    
    bvh_on_means = []
    bvh_off_means = []
    bvh_on_stds = []
    bvh_off_stds = []
    
    for scene in scenes:
        on_vals = df[(df['scene'] == scene) & (df['bvh'] == True)]['time'].values
        off_vals = df[(df['scene'] == scene) & (df['bvh'] == False)]['time'].values
        bvh_on_means.append(np.mean(on_vals) if len(on_vals) > 0 else np.nan)
        bvh_off_means.append(np.mean(off_vals) if len(off_vals) > 0 else np.nan)
        bvh_on_stds.append(np.std(on_vals) if len(on_vals) > 0 else np.nan)
        bvh_off_stds.append(np.std(off_vals) if len(off_vals) > 0 else np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bvh_on_means, width, yerr=bvh_on_stds, 
                   label='Build=1 (priemer)', color='green', capsize=5)
    bars2 = ax.bar(x + width/2, bvh_off_means, width, yerr=bvh_off_stds,
                   label='Build=0 (priemer)', color='red', capsize=5)
    
    ax.set_xlabel('Scéna')
    ax.set_ylabel('Priemerný čas (s)')
    ax.set_title('Vplyv Build na rýchlosť renderovania (priemer cez všetky kombinácie)', pad=35)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, bvh_on_means):
        if not np.isnan(val):
            ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, bvh_off_means):
        if not np.isnan(val):
            ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_line_by_combination(df, output_path='line_by_combination.png'):
    df['combination_idx'] = range(len(df))
    df['combination'] = df.apply(lambda row: 
        f"Build={int(row['bvh'])}, Fit={int(row['dynamic_bvh'])}, SIMD={int(row['simd'])}", axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    scenes = df['scene'].unique()
    for scene in scenes:
        scene_df = df[df['scene'] == scene]
        ax.plot(scene_df['combination_idx'], scene_df['time'], marker='o', label=scene, linewidth=2)
    
    ax.set_xlabel('Poradie kombinácií (podľa zoradenia)')
    ax.set_ylabel('Čas (s)')
    ax.set_title('Čas vykreslenia podľa poradia kombinácií', pad=35)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(scenes), frameon=False)
    
    ax.grid(True, alpha=0.3)
    
    all_combs = df['combination'].unique()
    step = max(1, len(all_combs)//10)
    xticks_pos = range(0, len(all_combs), step)
    xticks_labels = [all_combs[i] for i in xticks_pos]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_box_by_bvh(df, output_path='box_build.png'):
    data = [df[df['bvh'] == True]['time'].values,
            df[df['bvh'] == False]['time'].values]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, tick_labels=['Build=1', 'Build=0'], patch_artist=True)
    
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_ylabel('Čas (s)')
    ax.set_title('Porovnanie výkonu: Build=1 vs Build=0 (všetky kombinácie)', pad=35)
    
    green_patch = mpatches.Patch(color='lightgreen', label='Build=1')
    red_patch = mpatches.Patch(color='lightcoral', label='Build=0')
    ax.legend(handles=[green_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, (vals, color) in enumerate([(data[0], 'green'), (data[1], 'red')], start=1):
        x = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.5, color=color, s=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený ako {output_path}")

def main():
    try:
        df = load_benchmark_data()
        print("Načítané dáta:")
        print(df.head())
        
        plot_technique_comparison(df)
        plot_bvh_impact(df)
        plot_bvh_aggregated(df)
        plot_speedup(df)
        plot_line_by_combination(df)
        plot_box_by_bvh(df)
        plot_simd_comparison(df)
        
        print("Všetky grafy boli úspešne vygenerované.")
    except Exception as e:
        print(f"Chyba pri generovaní grafov: {e}")

if __name__ == "__main__":
    main()