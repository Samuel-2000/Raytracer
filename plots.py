import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_benchmark_data(csv_path='benchmark_results.csv'):
    """Načíta dáta z CSV súboru."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Súbor {csv_path} neexistuje. Spustite najprv benchmark.")
    return pd.read_csv(csv_path)

def plot_technique_comparison(df, output_path='benchmark_comparison.png'):
    """Vykreslí porovnanie techník pre každú scénu (stĺpcový graf)."""
    df['combination'] = df.apply(lambda row: 
        f"BVH={row['bvh']}, AS={row['adaptive']}, SS={row['subsampling']}, ND={row['neural']}", axis=1)
    
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
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_bvh_impact(df, output_path='bvh_impact.png'):
    """Vykreslí vplyv BVH na výkon pre každú scénu (stĺpcový graf)."""
    # Filtrujeme len kombinácie s vypnutými ostatnými technikami
    base = df[(df['adaptive'] == False) & (df['subsampling'] == False) & (df['neural'] == False)]
    
    scenes = base['scene'].unique()  # scény, ktoré majú aspoň jednu kombináciu v base
    x = np.arange(len(scenes))
    width = 0.35
    
    bvh_on = []
    bvh_off = []
    
    for scene in scenes:
        val_on = base[(base['scene'] == scene) & (base['bvh'] == True)]['time'].values
        val_off = base[(base['scene'] == scene) & (base['bvh'] == False)]['time'].values
        bvh_on.append(val_on[0] if len(val_on) > 0 else np.nan)
        bvh_off.append(val_off[0] if len(val_off) > 0 else np.nan)
    
    bvh_on = np.array(bvh_on)
    bvh_off = np.array(bvh_off)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bvh_on, width, label='BVH zapnuté', color='green')
    bars2 = ax.bar(x + width/2, bvh_off, width, label='BVH vypnuté', color='red')
    
    ax.set_xlabel('Scéna')
    ax.set_ylabel('Čas (s)')
    ax.set_title('Vplyv BVH na rýchlosť renderovania')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend()
    
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
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")


def plot_bvh_aggregated(df, output_path='bvh_aggregated.png'):
    """
    Stĺpcový graf – priemerný čas pre BVH True vs False naprieč všetkými kombináciami (AS, SS, ND).
    """
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
                   label='BVH zapnuté (priemer)', color='green', capsize=5)
    bars2 = ax.bar(x + width/2, bvh_off_means, width, yerr=bvh_off_stds,
                   label='BVH vypnuté (priemer)', color='red', capsize=5)
    
    ax.set_xlabel('Scéna')
    ax.set_ylabel('Priemerný čas (s)')
    ax.set_title('Vplyv BVH na rýchlosť renderovania (priemer cez všetky kombinácie)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Pridanie hodnôt nad stĺpce
    for bar, val in zip(bars1, bvh_on_means):
        if not np.isnan(val):
            ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, bvh_off_means):
        if not np.isnan(val):
            ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_speedup(df, output_path='speedup.png'):
    """Vykreslí zrýchlenie oproti základnej kombinácii (všetky tech. vypnuté)."""
    base_times = {}
    for scene in df['scene'].unique():
        base = df[(df['scene'] == scene) & 
                  (df['bvh'] == False) & 
                  (df['adaptive'] == False) & 
                  (df['subsampling'] == False) & 
                  (df['neural'] == False)]['time'].values
        base_times[scene] = base[0] if len(base) > 0 else np.nan
    
    df['speedup'] = df.apply(lambda row: base_times[row['scene']] / row['time'] if not np.isnan(base_times[row['scene']]) else np.nan, axis=1)
    df['combination'] = df.apply(lambda row: 
        f"BVH={row['bvh']}, AS={row['adaptive']}, SS={row['subsampling']}, ND={row['neural']}", axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    scenes = df['scene'].unique()
    for scene in scenes:
        scene_df = df[df['scene'] == scene]
        ax.plot(scene_df['combination'], scene_df['speedup'], marker='o', label=scene)
    
    ax.set_xlabel('Kombinácia techník')
    ax.set_ylabel('Zrýchlenie (násobok)')
    ax.set_title('Zrýchlenie oproti základnej konfigurácii (všetky techniky vypnuté)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_line_by_combination(df, output_path='line_by_combination.png'):
    """
    Line plot – čas pre každú scénu v poradí kombinácií (podľa CSV).
    Umožňuje sledovať trendy naprieč kombináciami.
    """
    df['combination_idx'] = range(len(df))
    df['combination'] = df.apply(lambda row: 
        f"BVH={row['bvh']}, AS={row['adaptive']}, SS={row['subsampling']}, ND={row['neural']}", axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    scenes = df['scene'].unique()
    for scene in scenes:
        scene_df = df[df['scene'] == scene]
        ax.plot(scene_df['combination_idx'], scene_df['time'], marker='o', label=scene, linewidth=2)
    
    ax.set_xlabel('Poradie kombinácií (podľa CSV)')
    ax.set_ylabel('Čas (s)')
    ax.set_title('Čas vykreslenia podľa poradia kombinácií (pre každú scénu)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Nastavíme xticks s názvami kombinácií (ale len niektoré, aby nebolo preplnené)
    all_combs = df['combination'].unique()
    step = max(1, len(all_combs)//10)
    xticks_pos = range(0, len(all_combs), step)
    xticks_labels = [all_combs[i] for i in xticks_pos]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")

def plot_box_by_bvh(df, output_path='box_bvh.png'):
    """
    Box plot – porovnanie časov pre BVH True vs False naprieč všetkými scénami a kombináciami.
    Zobrazuje distribúciu a mediány.
    """
    data = [df[df['bvh'] == True]['time'].values,
            df[df['bvh'] == False]['time'].values]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Použitie tick_labels namiesto labels (od matplotlib 3.9)
    bp = ax.boxplot(data, tick_labels=['BVH True', 'BVH False'], patch_artist=True)
    
    # Farbenie boxov
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_ylabel('Čas (s)')
    ax.set_title('Porovnanie výkonu s BVH a bez BVH (všetky kombinácie a scény)')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Pridanie bodov pre lepšiu predstavu
    for i, (vals, color) in enumerate([(data[0], 'green'), (data[1], 'red')], start=1):
        x = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.5, color=color, s=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Graf uložený ako {output_path}")




def main():
    """Hlavná funkcia na generovanie všetkých grafov."""
    try:
        df = load_benchmark_data()
        print("Načítané dáta:")
        print(df.head())
        
        # Základné grafy
        plot_technique_comparison(df)
        plot_bvh_impact(df)
        plot_bvh_aggregated(df)
        plot_speedup(df)
        
        # Nové grafy – line plot a box plot
        plot_line_by_combination(df)
        plot_box_by_bvh(df)

        
        print("Všetky grafy boli úspešne vygenerované.")
    except Exception as e:
        print(f"Chyba pri generovaní grafov: {e}")

if __name__ == "__main__":
    main()