import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PremierLeaguePredictorEnhanced:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.team_stats = {}
        self.historical_data = None
        self.current_season = None
        self.feature_names = []
        
    def load_data(self, file_paths):
        all_data = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            season = file_path.split('/')[-1].replace('Prem Results ', '').replace('Prem_Results_', '').replace('.csv', '')
            df['Season'] = season
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def parse_data(self, df):
        season_col = df['Season'].copy() if 'Season' in df.columns else None

        actual_cols = [col for col in df.columns if col != 'Season']
        
        if len(actual_cols) == 10:
            df.columns = ['Week', 'empty1', 'Date', 'empty2', 'Home', 'xG_Home', 'Score', 'xG_Away', 'Away', 'empty3'] + (['Season'] if 'Season' in df.columns else [])
        else:
            df.columns = ['Week', 'empty1', 'Date', 'empty2', 'Home', 'xG_Home', 'Score', 'xG_Away', 'Away'] + (['Season'] if 'Season' in df.columns else [])

        if season_col is not None and 'Season' not in df.columns:
            df['Season'] = season_col
        
        df = df[['Week', 'Date', 'Home', 'xG_Home', 'Score', 'xG_Away', 'Away', 'Season']].copy()
        
        df = df[df['Home'].notna()].copy()

        score_split = df['Score'].str.split('–', expand=True)
        if score_split.shape[1] >= 2:
            df['Goals_Home'] = score_split[0]
            df['Goals_Away'] = score_split[1]
        else:
            df['Goals_Home'] = None
            df['Goals_Away'] = None

        df['Goals_Home'] = pd.to_numeric(df['Goals_Home'], errors='coerce')
        df['Goals_Away'] = pd.to_numeric(df['Goals_Away'], errors='coerce')
        df['xG_Home'] = pd.to_numeric(df['xG_Home'], errors='coerce')
        df['xG_Away'] = pd.to_numeric(df['xG_Away'], errors='coerce')
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')

        df['Result'] = np.where(df['Goals_Home'] > df['Goals_Away'], 1,
                                np.where(df['Goals_Home'] < df['Goals_Away'], -1, 
                                        np.where(pd.isna(df['Goals_Home']), np.nan, 0)))

        df['Home'] = df['Home'].str.strip()
        df['Away'] = df['Away'].str.strip()
        
        return df
    
    def calculate_team_season_finish(self, df):
        season_finishes = {}
        
        for season in df['Season'].unique():
            season_df = df[df['Season'] == season].copy()
            
            team_points = {}
            for team in pd.concat([season_df['Home'], season_df['Away']]).unique():
                home_games = season_df[season_df['Home'] == team]
                home_points = (home_games['Result'] == 1).sum() * 3 + (home_games['Result'] == 0).sum()
                
                away_games = season_df[season_df['Away'] == team]
                away_points = (away_games['Result'] == -1).sum() * 3 + (away_games['Result'] == 0).sum()
                
                total_points = home_points + away_points
                
                goals_for = home_games['Goals_Home'].sum() + away_games['Goals_Away'].sum()
                goals_against = home_games['Goals_Away'].sum() + away_games['Goals_Home'].sum()
                goal_diff = goals_for - goals_against
                
                team_points[team] = {'points': total_points, 'goal_diff': goal_diff}
            
            sorted_teams = sorted(team_points.items(), 
                                key=lambda x: (x[1]['points'], x[1]['goal_diff']), 
                                reverse=True)
            
            season_finishes[season] = {team: idx + 1 for idx, (team, _) in enumerate(sorted_teams)}
        
        return season_finishes
    
    def get_previous_season_finish(self, team, season, season_finishes):
        season_24_25_positions = {
            'Liverpool': 1,
            'Arsenal': 2,
            'Manchester City': 3,
            'Chelsea': 4,
            'Newcastle United': 5,
            'Aston Villa': 6,
            'Nottingham Forest': 7,
            'Brighton': 8,
            'Bournemouth': 9,
            'Brentford': 10,
            'Fulham': 11,
            'Crystal Palace': 12,
            'Everton': 13,
            'West Ham United': 14,
            'Manchester Utd': 15,
            'Wolves': 16,
            'Tottenham Hotspur': 17
        }
        
        promoted_25_26 = ['Leeds United', 'Burnley', 'Sunderland']
        
        season_order = ['17_18', '18_19', '19_20', '20_21', '21_22', '22_23', '23_24', '24_25', '25_26']
        
        if season not in season_order:
            return 10  
        
        season_idx = season_order.index(season)
        if season_idx == 0:
            return 10  
        
        prev_season = season_order[season_idx - 1]
        
        if season == '25_26':
            if team in promoted_25_26:
                return 'Promoted 24/25'
            elif team in season_24_25_positions:
                return season_24_25_positions[team]
            else:
                return 10
        
        if prev_season in season_finishes and team in season_finishes[prev_season]:
            return season_finishes[prev_season][team]
        else:
            return 'Promoted'
        
    def calculate_form_features(self, df, team, match_date, match_week, n_matches=10, home_only=False, away_only=False):
        if home_only:
            team_matches = df[(df['Home'] == team) & (df['Week'] < match_week)].copy()
        elif away_only:
            team_matches = df[(df['Away'] == team) & (df['Week'] < match_week)].copy()
        else:
            team_matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Week'] < match_week)].copy()
        
        team_matches = team_matches.sort_values('Week').tail(n_matches)
        
        if len(team_matches) == 0:
            return {
                'points_per_game': 0,
                'goals_scored_avg': 0,
                'goals_conceded_avg': 0,
                'xg_for_avg': 0,
                'xg_against_avg': 0,
                'win_rate': 0,
                'matches_played': 0
            }
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        xg_for = 0
        xg_against = 0
        wins = 0
        
        for _, match in team_matches.iterrows():
            if match['Home'] == team:
                goals_scored += match['Goals_Home']
                goals_conceded += match['Goals_Away']
                xg_for += match['xG_Home']
                xg_against += match['xG_Away']
                
                if match['Result'] == 1:
                    points += 3
                    wins += 1
                elif match['Result'] == 0:
                    points += 1
            else:
                goals_scored += match['Goals_Away']
                goals_conceded += match['Goals_Home']
                xg_for += match['xG_Away']
                xg_against += match['xG_Home']
                
                if match['Result'] == -1:
                    points += 3
                    wins += 1
                elif match['Result'] == 0:
                    points += 1
        
        n = len(team_matches)
        
        return {
            'points_per_game': points / n,
            'goals_scored_avg': goals_scored / n,
            'goals_conceded_avg': goals_conceded / n,
            'xg_for_avg': xg_for / n,
            'xg_against_avg': xg_against / n,
            'win_rate': wins / n,
            'matches_played': n
        }
    
    def calculate_fixture_difficulty(self, df, team, match_week, opponent, n_matches=10):
        team_matches = df[((df['Home'] == team) | (df['Away'] == team)) & 
                         (df['Week'] < match_week)].copy()
        team_matches = team_matches.sort_values('Week').tail(n_matches)
        
        if len(team_matches) == 0:
            return 0.5
        
        opponent_strengths = []
        
        for _, match in team_matches.iterrows():
            opponent_name = match['Away'] if match['Home'] == team else match['Home']
            
            opp_form = self.calculate_form_features(df, opponent_name, match['Week'], match['Week'], n_matches=10)
            opponent_strengths.append(opp_form['points_per_game'])
        
        avg_opponent_strength = np.mean(opponent_strengths) if opponent_strengths else 1.5
        
        return min(avg_opponent_strength / 3, 1.0)
    
    def get_head_to_head_record(self, df, home_team, away_team, match_week):
        h2h_matches = df[
            (((df['Home'] == home_team) & (df['Away'] == away_team)) |
             ((df['Home'] == away_team) & (df['Away'] == home_team))) &
            (df['Week'] < match_week)
        ].copy()
        
        if len(h2h_matches) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_matches': 0,
                'h2h_home_goals_avg': 0,
                'h2h_away_goals_avg': 0
            }
        
        home_wins = ((h2h_matches['Home'] == home_team) & (h2h_matches['Result'] == 1)).sum()
        home_wins += ((h2h_matches['Away'] == home_team) & (h2h_matches['Result'] == -1)).sum()
        
        away_wins = ((h2h_matches['Home'] == away_team) & (h2h_matches['Result'] == 1)).sum()
        away_wins += ((h2h_matches['Away'] == away_team) & (h2h_matches['Result'] == -1)).sum()
        
        draws = (h2h_matches['Result'] == 0).sum()
        
        home_goals_when_home = h2h_matches[h2h_matches['Home'] == home_team]['Goals_Home'].mean()
        home_goals_when_away = h2h_matches[h2h_matches['Away'] == home_team]['Goals_Away'].mean()
        
        away_goals_when_away = h2h_matches[h2h_matches['Away'] == away_team]['Goals_Away'].mean()
        away_goals_when_home = h2h_matches[h2h_matches['Home'] == away_team]['Goals_Home'].mean()
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_matches': len(h2h_matches),
            'h2h_home_goals_avg': np.nanmean([home_goals_when_home, home_goals_when_away]),
            'h2h_away_goals_avg': np.nanmean([away_goals_when_away, away_goals_when_home])
        }
    
    def is_derby_match(self, home_team, away_team):
        derbies = [
            {'Manchester Utd', 'Manchester City'},
            {'Liverpool', 'Everton'},
            {'Arsenal', 'Tottenham Hotspur'},
            {'Chelsea', 'Arsenal'},
            {'Chelsea', 'Tottenham Hotspur'},
            {'Newcastle United', 'Sunderland'},
            {'Aston Villa', 'West Ham United'},
            {'Leeds United', 'Manchester Utd'},
            {'Liverpool', 'Manchester Utd'},
            {'Arsenal', 'Manchester Utd'},
        ]
        
        team_set = {home_team, away_team}
        for derby in derbies:
            if team_set == derby:
                return 1
        return 0
    
    def create_features(self, df, season_finishes):
        features_list = []
        
        for idx, row in df.iterrows():
            season_df = df[df['Season'] == row['Season']].copy()
            
            home_prev_finish = self.get_previous_season_finish(row['Home'], row['Season'], season_finishes)
            away_prev_finish = self.get_previous_season_finish(row['Away'], row['Season'], season_finishes)
            
            home_overall_form = self.calculate_form_features(season_df, row['Home'], row['Date'], row['Week'], n_matches=10)
            away_overall_form = self.calculate_form_features(season_df, row['Away'], row['Date'], row['Week'], n_matches=10)
            
            home_home_form = self.calculate_form_features(season_df, row['Home'], row['Date'], row['Week'], 
                                                         n_matches=10, home_only=True)
            
            away_away_form = self.calculate_form_features(season_df, row['Away'], row['Date'], row['Week'], 
                                                         n_matches=10, away_only=True)
            
            home_fixture_diff = self.calculate_fixture_difficulty(season_df, row['Home'], row['Week'], row['Away'])
            away_fixture_diff = self.calculate_fixture_difficulty(season_df, row['Away'], row['Week'], row['Home'])
            
            h2h = self.get_head_to_head_record(df, row['Home'], row['Away'], row['Week'])
            
            home_promoted = 1 if (isinstance(home_prev_finish, str) and 'Promoted' in home_prev_finish) or (isinstance(home_prev_finish, (int, float)) and home_prev_finish >= 18) else 0
            away_promoted = 1 if (isinstance(away_prev_finish, str) and 'Promoted' in away_prev_finish) or (isinstance(away_prev_finish, (int, float)) and away_prev_finish >= 18) else 0
            
            home_prev_finish_numeric = 21 if isinstance(home_prev_finish, str) and 'Promoted' in home_prev_finish else home_prev_finish
            away_prev_finish_numeric = 21 if isinstance(away_prev_finish, str) and 'Promoted' in away_prev_finish else away_prev_finish
            
            feature_dict = {
                'home_prev_finish': home_prev_finish_numeric,
                'away_prev_finish': away_prev_finish_numeric,
                'home_promoted': home_promoted,
                'away_promoted': away_promoted,
                
                'home_overall_ppg': home_overall_form['points_per_game'],
                'home_overall_gf': home_overall_form['goals_scored_avg'],
                'home_overall_ga': home_overall_form['goals_conceded_avg'],
                'home_overall_xg': home_overall_form['xg_for_avg'],
                'home_overall_xga': home_overall_form['xg_against_avg'],
                'home_overall_winrate': home_overall_form['win_rate'],
                
                'away_overall_ppg': away_overall_form['points_per_game'],
                'away_overall_gf': away_overall_form['goals_scored_avg'],
                'away_overall_ga': away_overall_form['goals_conceded_avg'],
                'away_overall_xg': away_overall_form['xg_for_avg'],
                'away_overall_xga': away_overall_form['xg_against_avg'],
                'away_overall_winrate': away_overall_form['win_rate'],
                
                'home_home_ppg': home_home_form['points_per_game'],
                'home_home_gf': home_home_form['goals_scored_avg'],
                'home_home_ga': home_home_form['goals_conceded_avg'],
                'home_home_xg': home_home_form['xg_for_avg'],
                'home_home_xga': home_home_form['xg_against_avg'],
                'home_home_winrate': home_home_form['win_rate'],
                
                'away_away_ppg': away_away_form['points_per_game'],
                'away_away_gf': away_away_form['goals_scored_avg'],
                'away_away_ga': away_away_form['goals_conceded_avg'],
                'away_away_xg': away_away_form['xg_for_avg'],
                'away_away_xga': away_away_form['xg_against_avg'],
                'away_away_winrate': away_away_form['win_rate'],
                
                'home_fixture_difficulty': home_fixture_diff,
                'away_fixture_difficulty': away_fixture_diff,
                
                'h2h_home_wins': h2h['h2h_home_wins'],
                'h2h_draws': h2h['h2h_draws'],
                'h2h_away_wins': h2h['h2h_away_wins'],
                'h2h_matches': h2h['h2h_matches'],
                'h2h_home_goals': h2h['h2h_home_goals_avg'],
                'h2h_away_goals': h2h['h2h_away_goals_avg'],
                
                'is_derby': self.is_derby_match(row['Home'], row['Away']),
                
                'result': row['Result']
            }
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def train_models(self, X_train, y_train):
        print("Training models...")
        
        y_train_cat = y_train + 1
        
        print("  - Logistic Regression")
        self.models['logistic'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['logistic'].fit(X_train, y_train_cat)
        
        print("  - Random Forest")
        self.models['random_forest'] = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                                              random_state=42, n_jobs=-1)
        self.models['random_forest'].fit(X_train, y_train_cat)
        
        print("  - XGBoost")
        self.models['xgboost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                              random_state=42, eval_metric='mlogloss')
        self.models['xgboost'].fit(X_train, y_train_cat)
        
        print("Training complete!\n")
    
    def evaluate_models(self, X_test, y_test):
        y_test_cat = y_test + 1
        
        print("="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}")
            print("-"*70)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test_cat, y_pred)
            logloss = log_loss(y_test_cat, y_pred_proba)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Log Loss: {logloss:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test_cat, y_pred, 
                                      target_names=['Away Win', 'Draw', 'Home Win'],
                                      zero_division=0))
    
    def get_feature_importance(self, model_name='random_forest'):
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None
    
    def calibrate_draw_probability(self, probas):
        away_win, draw, home_win = probas[0], probas[1], probas[2]
        
        draw_boost = 1.15
        draw_adjusted = min(draw * draw_boost, 0.45)
        
        total_wins = away_win + home_win
        if total_wins > 0:
            reduction_factor = (1 - draw_adjusted) / total_wins
            away_win_adjusted = away_win * reduction_factor
            home_win_adjusted = home_win * reduction_factor
        else:
            away_win_adjusted = (1 - draw_adjusted) / 2
            home_win_adjusted = (1 - draw_adjusted) / 2
        
        total = away_win_adjusted + draw_adjusted + home_win_adjusted
        
        return np.array([
            away_win_adjusted / total,
            draw_adjusted / total,
            home_win_adjusted / total
        ])
    
    def create_combined_visualization(self, match_figures, output_file):
        n_matches = len(match_figures)
        
        n_cols = 2
        n_rows = (n_matches + 1) // 2
        
        fig = plt.figure(figsize=(32, 12 * n_rows))
        
        for idx, match_data in enumerate(match_figures):
            base_idx = idx * 4
            
            row = idx // n_cols
            col = idx % n_cols
            
            gs = fig.add_gridspec(n_rows * 2, n_cols * 2, 
                                 left=0.05, right=0.95, 
                                 top=0.95, bottom=0.05,
                                 hspace=0.4, wspace=0.3)
            
            row_start = row * 2
            col_start = col * 2
            
            ax_title = fig.add_subplot(gs[row_start:row_start+2, col_start:col_start+2])
            ax_title.text(0.5, 0.95, f'{match_data["home"]} vs {match_data["away"]}', 
                         ha='center', va='top', fontsize=20, fontweight='bold',
                         transform=ax_title.transAxes)
            ax_title.axis('off')
            
            self.create_match_panels(fig, gs, row_start, col_start, 
                                   match_data['home'], match_data['away'],
                                   match_data['features'], match_data['probas'])
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_match_panels(self, fig, gs, row_start, col_start, home_team, away_team, features, probabilities):
        ax1 = fig.add_subplot(gs[row_start, col_start])
        outcomes = ['Away Win', 'Draw', 'Home Win']
        colors = ['#e74c3c', '#95a5a6', '#3498db']
        bars = ax1.barh(outcomes, probabilities[[0, 1, 2]], color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Probability', fontweight='bold', fontsize=10)
        ax1.set_title('Match Outcome Probabilities', fontweight='bold', fontsize=12)
        ax1.set_xlim([0, 1])
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities[[0, 1, 2]])):
            ax1.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        max_prob = max(probabilities)
        if max_prob > 0.6:
            confidence = "HIGH"
            conf_color = "green"
        elif max_prob > 0.45:
            confidence = "MEDIUM"
            conf_color = "orange"
        else:
            confidence = "LOW"
            conf_color = "red"
        
        ax1.text(0.5, -0.15, f'Confidence: {confidence}', 
                transform=ax1.transAxes, ha='center', fontsize=11, 
                fontweight='bold', color=conf_color)
        
        ax2 = fig.add_subplot(gs[row_start, col_start + 1])
        categories = ['Overall\nPPG', 'Home/Away\nPPG', 'Goals\nScored', 'Goals\nConceded', 'xG For', 'xG Against']
        home_values = [
            features['home_overall_ppg'],
            features['home_home_ppg'],
            features['home_overall_gf'],
            features['home_overall_ga'],
            features['home_overall_xg'],
            features['home_overall_xga']
        ]
        away_values = [
            features['away_overall_ppg'],
            features['away_away_ppg'],
            features['away_overall_gf'],
            features['away_overall_ga'],
            features['away_overall_xg'],
            features['away_overall_xga']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, home_values, width, label=home_team[:15], color='#3498db', alpha=0.7, edgecolor='black')
        ax2.bar(x + width/2, away_values, width, label=away_team[:15], color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax2.set_ylabel('Value', fontweight='bold', fontsize=10)
        ax2.set_title('Recent Form Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        ax3 = fig.add_subplot(gs[row_start + 1, col_start])
        if features['h2h_matches'] > 0:
            h2h_data = [features['h2h_home_wins'], features['h2h_draws'], features['h2h_away_wins']]
            h2h_labels = [f'{home_team[:12]}\nWins', 'Draws', f'{away_team[:12]}\nWins']
            colors_h2h = ['#3498db', '#95a5a6', '#e74c3c']
            
            wedges, texts, autotexts = ax3.pie(h2h_data, labels=h2h_labels, autopct='%1.0f%%',
                                               colors=colors_h2h, startangle=90, 
                                               textprops={'fontweight': 'bold', 'fontsize': 9})
            ax3.set_title(f'Head-to-Head ({int(features["h2h_matches"])} matches)', fontweight='bold', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'No Previous\nMeetings', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12, fontweight='bold')
            ax3.set_title('Head-to-Head Record', fontweight='bold', fontsize=12)
            ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row_start + 1, col_start + 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        home_prev_display = features.get('home_prev_finish_display', features['home_prev_finish'])
        away_prev_display = features.get('away_prev_finish_display', features['away_prev_finish'])
        
        home_prev_str = home_prev_display if isinstance(home_prev_display, str) else f'{int(home_prev_display)}'
        away_prev_str = away_prev_display if isinstance(away_prev_display, str) else f'{int(away_prev_display)}'
        
        home_prev_str = home_prev_str.replace('Promoted', 'Prom') if isinstance(home_prev_str, str) else home_prev_str
        away_prev_str = away_prev_str.replace('Promoted', 'Prom') if isinstance(away_prev_str, str) else away_prev_str
        
        if isinstance(home_prev_display, str) and isinstance(away_prev_display, str):
            prev_season_adv = 'Shared'
        elif isinstance(home_prev_display, str):
            prev_season_adv = away_team[:10]
        elif isinstance(away_prev_display, str):
            prev_season_adv = home_team[:10]
        else:
            prev_season_adv = home_team[:10] if home_prev_display < away_prev_display else away_team[:10]
        
        table_data = [
            ['Metric', home_team[:12], away_team[:12], 'Advantage'],
            ['Prev Season', home_prev_str, away_prev_str, prev_season_adv],
            ['Overall PPG', f'{features["home_overall_ppg"]:.3f}', 
             f'{features["away_overall_ppg"]:.3f}',
             home_team[:10] if features['home_overall_ppg'] > features['away_overall_ppg'] else away_team[:10]],
            ['Home/Away PPG', f'{features["home_home_ppg"]:.3f}', 
             f'{features["away_away_ppg"]:.3f}',
             home_team[:10] if features['home_home_ppg'] > features['away_away_ppg'] else away_team[:10]],
            ['xG For', f'{features["home_overall_xg"]:.3f}', 
             f'{features["away_overall_xg"]:.3f}',
             home_team[:10] if features['home_overall_xg'] > features['away_overall_xg'] else away_team[:10]],
            ['Goals Conc.', f'{features["home_overall_ga"]:.3f}', 
             f'{features["home_overall_ga"]:.3f}',
             home_team[:10] if features['home_overall_ga'] < features['away_overall_ga'] else away_team[:10]],
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax4.set_title('Key Statistics', fontweight='bold', fontsize=12, pad=10)
    
    def create_match_visualization(self, home_team, away_team, features, probabilities, output_file=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{home_team} vs {away_team} - Detailed Analysis', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        outcomes = ['Away Win', 'Draw', 'Home Win']
        colors = ['#e74c3c', '#95a5a6', '#3498db']
        bars = ax1.barh(outcomes, probabilities[[0, 1, 2]], color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Probability', fontweight='bold')
        ax1.set_title('Match Outcome Probabilities', fontweight='bold')
        ax1.set_xlim([0, 1])
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities[[0, 1, 2]])):
            ax1.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontweight='bold')
        
        max_prob = max(probabilities)
        if max_prob > 0.6:
            confidence = "HIGH"
            conf_color = "green"
        elif max_prob > 0.45:
            confidence = "MEDIUM"
            conf_color = "orange"
        else:
            confidence = "LOW"
            conf_color = "red"
        
        ax1.text(0.5, -0.5, f'Prediction Confidence: {confidence}', 
                transform=ax1.transAxes, ha='center', fontsize=12, 
                fontweight='bold', color=conf_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2 = axes[0, 1]
        categories = ['Overall\nPPG', 'Home/Away\nPPG', 'Goals\nScored', 'Goals\nConceded', 'xG For', 'xG Against']
        home_values = [
            features['home_overall_ppg'],
            features['home_home_ppg'],
            features['home_overall_gf'],
            features['home_overall_ga'],
            features['home_overall_xg'],
            features['home_overall_xga']
        ]
        away_values = [
            features['away_overall_ppg'],
            features['away_away_ppg'],
            features['away_overall_gf'],
            features['away_overall_ga'],
            features['away_overall_xg'],
            features['away_overall_xga']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, home_values, width, label=home_team, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.bar(x + width/2, away_values, width, label=away_team, color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Recent Form Comparison (Last 5 Matches)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        ax3 = axes[1, 0]
        if features['h2h_matches'] > 0:
            h2h_data = [features['h2h_home_wins'], features['h2h_draws'], features['h2h_away_wins']]
            h2h_labels = [f'{home_team}\nWins', 'Draws', f'{away_team}\nWins']
            colors_h2h = ['#3498db', '#95a5a6', '#e74c3c']
            
            wedges, texts, autotexts = ax3.pie(h2h_data, labels=h2h_labels, autopct='%1.0f%%',
                                               colors=colors_h2h, startangle=90, textprops={'fontweight': 'bold'})
            ax3.set_title(f'Head-to-Head Record ({int(features["h2h_matches"])} matches)', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Previous\nMeetings', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, fontweight='bold')
            ax3.set_title('Head-to-Head Record', fontweight='bold')
            ax3.axis('off')
        
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        home_prev_display = features.get('home_prev_finish_display', features['home_prev_finish'])
        away_prev_display = features.get('away_prev_finish_display', features['away_prev_finish'])
        
        home_prev_str = home_prev_display if isinstance(home_prev_display, str) else f'{int(home_prev_display)}'
        away_prev_str = away_prev_display if isinstance(away_prev_display, str) else f'{int(away_prev_display)}'
        
        if isinstance(home_prev_display, str) and isinstance(away_prev_display, str):
            prev_season_adv = 'Shared'
        elif isinstance(home_prev_display, str):
            prev_season_adv = away_team
        elif isinstance(away_prev_display, str):
            prev_season_adv = home_team
        else:
            prev_season_adv = home_team if home_prev_display < away_prev_display else away_team
        
        table_data = [
            ['Metric', home_team, away_team, 'Advantage'],
            ['Previous Season Finish', home_prev_str, away_prev_str, prev_season_adv],
            ['Overall Form (PPG)', f'{features["home_overall_ppg"]:.3f}', 
             f'{features["away_overall_ppg"]:.3f}',
             home_team if features['home_overall_ppg'] > features['away_overall_ppg'] else away_team],
            ['Home/Away Form (PPG)', f'{features["home_home_ppg"]:.3f}', 
             f'{features["away_away_ppg"]:.3f}',
             home_team if features['home_home_ppg'] > features['away_away_ppg'] else away_team],
            ['Expected Goals (xG)', f'{features["home_overall_xg"]:.3f}', 
             f'{features["away_overall_xg"]:.3f}',
             home_team if features['home_overall_xg'] > features['away_overall_xg'] else away_team],
            ['Goals Conceded', f'{features["home_overall_ga"]:.3f}', 
             f'{features["away_overall_ga"]:.3f}',
             home_team if features['home_overall_ga'] < features['away_overall_ga'] else away_team],
            ['Fixture Difficulty', f'{features["home_fixture_difficulty"]:.3f}', 
             f'{features["away_fixture_difficulty"]:.3f}',
             'Similar' if abs(features['home_fixture_difficulty'] - features['away_fixture_difficulty']) < 0.1 
             else (home_team if features['home_fixture_difficulty'] > features['away_fixture_difficulty'] else away_team)],
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax4.set_title('Key Statistics Comparison', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
    
    def predict_matches(self, matches_df, all_data, season_finishes, model_name='random_forest', create_visuals=True):
        import os
        predictions = []
        match_figures = []
        
        if create_visuals:
            os.makedirs('gwk23_pngs', exist_ok=True)
        
        for idx, match in matches_df.iterrows():
            season_df = all_data[all_data['Season'] == match['Season']].copy()
            
            home_prev_finish = self.get_previous_season_finish(match['Home'], match['Season'], season_finishes)
            away_prev_finish = self.get_previous_season_finish(match['Away'], match['Season'], season_finishes)
            
            home_overall_form = self.calculate_form_features(season_df, match['Home'], match['Date'], 
                                                            match['Week'], n_matches=10)
            away_overall_form = self.calculate_form_features(season_df, match['Away'], match['Date'], 
                                                            match['Week'], n_matches=10)
            
            home_home_form = self.calculate_form_features(season_df, match['Home'], match['Date'], 
                                                         match['Week'], n_matches=10, home_only=True)
            away_away_form = self.calculate_form_features(season_df, match['Away'], match['Date'], 
                                                         match['Week'], n_matches=10, away_only=True)
            
            home_fixture_diff = self.calculate_fixture_difficulty(season_df, match['Home'], match['Week'], match['Away'])
            away_fixture_diff = self.calculate_fixture_difficulty(season_df, match['Away'], match['Week'], match['Home'])
            
            h2h = self.get_head_to_head_record(all_data, match['Home'], match['Away'], match['Week'])
            
            home_prev_finish_display = home_prev_finish
            away_prev_finish_display = away_prev_finish
            
            home_promoted = 1 if (isinstance(home_prev_finish, str) and 'Promoted' in home_prev_finish) or (isinstance(home_prev_finish, (int, float)) and home_prev_finish >= 18) else 0
            away_promoted = 1 if (isinstance(away_prev_finish, str) and 'Promoted' in away_prev_finish) or (isinstance(away_prev_finish, (int, float)) and away_prev_finish >= 18) else 0
            
            home_prev_finish_numeric = 21 if isinstance(home_prev_finish, str) and 'Promoted' in home_prev_finish else home_prev_finish
            away_prev_finish_numeric = 21 if isinstance(away_prev_finish, str) and 'Promoted' in away_prev_finish else away_prev_finish
            
            features_for_model = {
                'home_prev_finish': home_prev_finish_numeric,
                'away_prev_finish': away_prev_finish_numeric,
                'home_promoted': home_promoted,
                'away_promoted': away_promoted,
                'home_overall_ppg': home_overall_form['points_per_game'],
                'home_overall_gf': home_overall_form['goals_scored_avg'],
                'home_overall_ga': home_overall_form['goals_conceded_avg'],
                'home_overall_xg': home_overall_form['xg_for_avg'],
                'home_overall_xga': home_overall_form['xg_against_avg'],
                'home_overall_winrate': home_overall_form['win_rate'],
                'away_overall_ppg': away_overall_form['points_per_game'],
                'away_overall_gf': away_overall_form['goals_scored_avg'],
                'away_overall_ga': away_overall_form['goals_conceded_avg'],
                'away_overall_xg': away_overall_form['xg_for_avg'],
                'away_overall_xga': away_overall_form['xg_against_avg'],
                'away_overall_winrate': away_overall_form['win_rate'],
                'home_home_ppg': home_home_form['points_per_game'],
                'home_home_gf': home_home_form['goals_scored_avg'],
                'home_home_ga': home_home_form['goals_conceded_avg'],
                'home_home_xg': home_home_form['xg_for_avg'],
                'home_home_xga': home_home_form['xg_against_avg'],
                'home_home_winrate': home_home_form['win_rate'],
                'away_away_ppg': away_away_form['points_per_game'],
                'away_away_gf': away_away_form['goals_scored_avg'],
                'away_away_ga': away_away_form['goals_conceded_avg'],
                'away_away_xg': away_away_form['xg_for_avg'],
                'away_away_xga': away_away_form['xg_against_avg'],
                'away_away_winrate': away_away_form['win_rate'],
                'home_fixture_difficulty': home_fixture_diff,
                'away_fixture_difficulty': away_fixture_diff,
                'h2h_home_wins': h2h['h2h_home_wins'],
                'h2h_draws': h2h['h2h_draws'],
                'h2h_away_wins': h2h['h2h_away_wins'],
                'h2h_matches': h2h['h2h_matches'],
                'h2h_home_goals': h2h['h2h_home_goals_avg'],
                'h2h_away_goals': h2h['h2h_away_goals_avg'],
                'is_derby': self.is_derby_match(match['Home'], match['Away'])
            }
            
            features_for_viz = {**features_for_model}
            features_for_viz['home_prev_finish_display'] = home_prev_finish_display
            features_for_viz['away_prev_finish_display'] = away_prev_finish_display
            
            X = pd.DataFrame([features_for_model])
            X_scaled = self.scaler.transform(X)
            
            probas_raw = self.models[model_name].predict_proba(X_scaled)[0]
            
            probas = self.calibrate_draw_probability(probas_raw)
            
            prediction = int(np.argmax(probas))
            
            max_prob = max(probas)
            if max_prob > 0.6:
                confidence = "HIGH"
            elif max_prob > 0.45:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            if create_visuals:
                match_figures.append({
                    'home': match['Home'],
                    'away': match['Away'],
                    'features': features_for_viz,
                    'probas': probas
                })
                
                output_filename = f"gwk23_pngs/match_analysis_{match['Home'].replace(' ', '_')}_vs_{match['Away'].replace(' ', '_')}.png"
                self.create_match_visualization(match['Home'], match['Away'], features_for_viz, probas, output_file=output_filename)
            
            predictions.append({
                'Home': match['Home'],
                'Away': match['Away'],
                'Predicted_Result': ['Away Win', 'Draw', 'Home Win'][prediction],
                'Home_Win_Prob': round(probas[2], 3),
                'Draw_Prob': round(probas[1], 3),
                'Away_Win_Prob': round(probas[0], 3),
                'Confidence': confidence,
                'Home_Form_PPG': round(home_overall_form['points_per_game'], 3),
                'Away_Form_PPG': round(away_overall_form['points_per_game'], 3),
                'Home_Home_PPG': round(home_home_form['points_per_game'], 3),
                'Away_Away_PPG': round(away_away_form['points_per_game'], 3),
                'Home_xG': round(home_overall_form['xg_for_avg'], 3),
                'Away_xG': round(away_overall_form['xg_for_avg'], 3),
                'Home_xGA': round(home_overall_form['xg_against_avg'], 3),
                'Away_xGA': round(away_overall_form['xg_against_avg'], 3),
                'H2H_Matches': h2h['h2h_matches']
            })
        
        if create_visuals and len(match_figures) > 0:
            self.create_combined_visualization(match_figures, 'all_matches_analysis.png')
            print(f"\n📊 All match analysis charts saved to: all_matches_analysis.png")
            print(f"📊 Individual match PNGs saved to: gwk23_pngs/ folder")
        
        return pd.DataFrame(predictions)

def main():
    print("="*70)
    print("PREMIER LEAGUE MATCH PREDICTOR - ENHANCED VERSION")
    print("="*70)
    print()
    
    predictor = PremierLeaguePredictorEnhanced()
    
    print("Loading historical data...")
    historical_files = [
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 17_18.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 18_19.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 19_20.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 20_21.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 21_22.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 22_23.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 23_24.csv',
        '/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 24_25.csv'
    ]
    
    historical_data = predictor.load_data(historical_files)
    historical_data = predictor.parse_data(historical_data)
    print(f"Loaded {len(historical_data)} historical matches")
    
    print("Loading current season data...")
    current_season_df = pd.read_csv('/Users/hasanshariff/Desktop/gameweek_predictor/data/Prem Results 25_26.csv', encoding='utf-8-sig')
    current_season_df['Season'] = '25_26'
    current_season_df = predictor.parse_data(current_season_df)
    
    completed_current = current_season_df[current_season_df['Score'].notna()].copy()
    upcoming_matches = current_season_df[current_season_df['Score'].isna()].copy()
    
    print(f"Current season: {len(completed_current)} completed, {len(upcoming_matches)} upcoming matches")
    print()
    
    all_data = pd.concat([historical_data, completed_current], ignore_index=True)
    
    print("Calculating season standings...")
    season_finishes = predictor.calculate_team_season_finish(all_data)
    
    print("Creating features...")
    feature_df = predictor.create_features(all_data, season_finishes)
    predictor.feature_names = [col for col in feature_df.columns if col != 'result']
    
    print(f"Created {len(feature_df)} training samples with {len(feature_df.columns)-1} features")
    print()
    
    X = feature_df[feature_df['result'].notna()].drop('result', axis=1)
    y = feature_df[feature_df['result'].notna()]['result']
    
    X_scaled = predictor.scaler.fit_transform(X)
    
    split_point = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    predictor.train_models(X_train, y_train)
    
    predictor.evaluate_models(X_test, y_test)
    
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    feature_importance = predictor.get_feature_importance('random_forest')
    if feature_importance is not None:
        print(feature_importance.head(10).to_string(index=False))
    print()
    
    if len(upcoming_matches) > 0:
        print("\n" + "="*70)
        print("PREDICTIONS FOR UPCOMING MATCHES")
        print("="*70)
        print()
        
        predictions = predictor.predict_matches(upcoming_matches, all_data, season_finishes, 
                                               model_name='random_forest', create_visuals=True)
        
        for idx, pred in predictions.iterrows():
            print(f"\n{'='*70}")
            print(f"{pred['Home']} vs {pred['Away']}")
            print(f"{'='*70}")
            print(f"PREDICTION: {pred['Predicted_Result']} (Confidence: {pred['Confidence']})")
            print(f"\nProbabilities:")
            print(f"  Home Win: {pred['Home_Win_Prob']:.1%}")
            print(f"  Draw:     {pred['Draw_Prob']:.1%}")
            print(f"  Away Win: {pred['Away_Win_Prob']:.1%}")
            print(f"\nRecent Form (Last 5 Matches):")
            print(f"  {pred['Home']:20s} - Overall: {pred['Home_Form_PPG']:.3f} PPG | Home: {pred['Home_Home_PPG']:.3f} PPG")
            print(f"  {pred['Away']:20s} - Overall: {pred['Away_Form_PPG']:.3f} PPG | Away: {pred['Away_Away_PPG']:.3f} PPG")
            print(f"\nExpected Goals:")
            print(f"  {pred['Home']:20s} - xG For: {pred['Home_xG']:.3f} | xG Against: {pred['Home_xGA']:.3f}")
            print(f"  {pred['Away']:20s} - xG For: {pred['Away_xG']:.3f} | xG Against: {pred['Away_xGA']:.3f}")
            
            if pred['H2H_Matches'] > 0:
                print(f"\nHead-to-Head: {int(pred['H2H_Matches'])} previous meetings")
            else:
                print(f"\nHead-to-Head: No previous meetings")
        
        predictions.to_csv('gameweek23_predictions_detailed.csv', index=False)
        print("\n" + "="*70)
        print("Predictions saved to: gameweek_predictions_detailed.csv")
        print("All match visualizations saved to: all_matches_analysis.png")
        print("="*70)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()