import os
import json
import pandas as pd
from datetime import datetime

# 설정
DATA_DIR = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\PUBG_data'
OUTPUT_DIR = 'output'

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(file_path):
    """JSON 파일을 로드하여 파이썬 객체로 반환."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"File Load Error: {file_path}\nError: {e}")
        return None

def parse_iso8601(timestamp_str):
    """ISO 8601 형식의 타임스탬프 문자열을 datetime 객체로 변환."""
    try:
        # 마이크로초 부분 조정
        if '.' in timestamp_str:
            # '.'과 'Z'의 위치 찾기
            dot_index = timestamp_str.find('.')
            z_index = timestamp_str.find('Z', dot_index)
            if z_index == -1:
                z_index = len(timestamp_str)
            microseconds = timestamp_str[dot_index+1:z_index]
            # 마이크로초를 6자리로 조정 (필요 시 자르거나 패딩)
            if len(microseconds) > 6:
                microseconds = microseconds[:6]
            else:
                microseconds = microseconds.ljust(6, '0')
            # 조정된 타임스탬프 문자열 생성
            timestamp_str = timestamp_str[:dot_index+1] + microseconds + timestamp_str[z_index:]
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            print(f"Timestamp parsing error: {timestamp_str}")
            return None

def extract_item_usage(telemetry_data, account_id):
    """
    텔레메트리 데이터에서 아이템 사용 정보를 추출.

    Args:
        telemetry_data (list): 텔레메트리 이벤트 리스트.
        account_id (str): 플레이어의 accountId.

    Returns:
        dict: 아이템 사용 관련 정보.
    """
    primary_weapon = None
    secondary_weapon = None
    armor_type = "None"
    health_items_used = 0
    boost_items_used = 0

    for event in telemetry_data:
        # 아이템 장착 이벤트 처리
        if event.get('_T') == 'LogItemEquip' and event.get('character', {}).get('accountId') == account_id:
            item_category = event.get('item', {}).get('category', None)
            item_id = event.get('item', {}).get('itemId', None)
            if item_category == 'Weapon':
                if not primary_weapon:
                    primary_weapon = item_id
                elif not secondary_weapon:
                    secondary_weapon = item_id
            elif item_category == 'Armor':
                armor_type = item_id

        # 아이템 사용 이벤트 처리
        if event.get('_T') == 'LogItemUse' and event.get('character', {}).get('accountId') == account_id:
            item_category = event.get('item', {}).get('category', None)
            if item_category == 'Healing':
                health_items_used += 1
            elif item_category == 'Boost':
                boost_items_used += 1

    return {
        'primary_weapon': primary_weapon,
        'secondary_weapon': secondary_weapon,
        'armor_type': armor_type,
        'use_of_health_items': health_items_used,
        'use_of_boost_items': boost_items_used
    }

def extract_movement_routes(telemetry_data, account_id, match_start_time):
    """
    텔레메트리 데이터에서 특정 플레이어의 이동 경로를 추출.

    Args:
        telemetry_data (list): 텔레메트리 이벤트 리스트.
        account_id (str): 플레이어의 accountId.
        match_start_time (datetime): 매치 시작 시간.

    Returns:
        list: 시간순으로 정렬된 (relative_seconds, x, y, z) 튜플 리스트.
    """
    movement_routes = []

    # LogPlayerPosition 이벤트 처리
    player_position_events = [
        event for event in telemetry_data
        if event.get('_T') == 'LogPlayerPosition' and event.get('character', {}).get('accountId') == account_id
    ]

    for event in player_position_events:
        loc = event.get('character', {}).get('location', {})
        x = loc.get('x', None)
        y = loc.get('y', None)
        z = loc.get('z', None)
        timestamp_str = event.get('_D', None)
        timestamp_dt = parse_iso8601(timestamp_str)
        if timestamp_dt and match_start_time:
            relative_seconds = (timestamp_dt - match_start_time).total_seconds()
            if x is not None and y is not None and z is not None:
                movement_routes.append((relative_seconds, x, y, z))

    # 이동 경로 정렬
    movement_routes.sort(key=lambda x: x[0])  # 상대 타임스탬프 기준 정렬
    return movement_routes

def extract_additional_data(telemetry_data, account_id, players_stats):
    """
    텔레메트리 데이터에서 추가 정보를 추출.

    Args:
        telemetry_data (list): 텔레메트리 이벤트 리스트.
        account_id (str): 플레이어의 accountId.
        players_stats (dict): 플레이어의 stats 정보 (players.json에서 추출).

    Returns:
        dict: 추가 정보.
    """
    items_carried = []
    loot_events = []
    combat_events = []

    for event in telemetry_data:
        # 아이템 픽업 이벤트 처리
        if event.get('_T') == 'LogItemPickup' and event.get('character', {}).get('accountId') == account_id:
            item = event.get('item', {}).get('itemId', None)
            if item:
                items_carried.append(item)
                loot_events.append(event.get('_D', None))

        # 플레이어 공격 이벤트 처리
        if event.get('_T') == 'LogPlayerAttack' and event.get('attacker', {}).get('accountId') == account_id:
            combat_events.append(event.get('_D', None))

    # 로팅(looting) 시간 계산
    time_spent_looting = 0
    if loot_events:
        loot_timestamps = [parse_iso8601(ts) for ts in loot_events if ts]
        loot_timestamps = [ts for ts in loot_timestamps if ts is not None]
        if loot_timestamps and len(loot_timestamps) > 1:
            time_spent_looting = (max(loot_timestamps) - min(loot_timestamps)).total_seconds()

    # 전투(combat) 시간 계산
    time_spent_in_combat = 0
    if combat_events:
        combat_timestamps = [parse_iso8601(ts) for ts in combat_events if ts]
        combat_timestamps = [ts for ts in combat_timestamps if ts is not None]
        if combat_timestamps and len(combat_timestamps) > 1:
            time_spent_in_combat = (max(combat_timestamps) - min(combat_timestamps)).total_seconds()

    kills = players_stats.get('kills', 0)
    damage_dealt = players_stats.get('damageDealt', 0)

    return {
        'items_carried': ', '.join(items_carried) if items_carried else "None",
        'time_spent_looting_sec': time_spent_looting,
        'time_spent_in_combat_sec': time_spent_in_combat,
        'kills': kills,
        'damage_dealt': damage_dealt
    }

def extract_team_info(roster_data):
    """
    roster.json 데이터에서 팀 정보를 추출.

    Args:
        roster_data (list): roster.json 이벤트 리스트.

    Returns:
        dict: teamId를 키로 하는 팀 정보 딕셔너리.
    """
    team_info = {}
    for roster in roster_data:
        if roster.get('type') != 'roster':
            continue
        attributes = roster.get('attributes', {})
        stats = attributes.get('stats', {})
        rank = stats.get('rank', None)
        team_id = stats.get('teamId', None)
        won = attributes.get('won', "false")  # 문자열 "true" 또는 "false"

        team_info[team_id] = {
            'team_rank': rank,
            'team_won': won
        }
    return team_info

def compile_player_data(data_dir):
    """
    모든 플레이어 정보를 추출하여 데이터프레임 생성.

    Args:
        data_dir (str): 데이터 디렉토리 경로.

    Returns:
        pd.DataFrame: 플레이어 데이터프레임.
    """
    player_data = []

    for user_id in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_id)
        if not os.path.isdir(user_path):
            continue

        for match_id in os.listdir(user_path):
            match_path = os.path.join(user_path, match_id)
            if not os.path.isdir(match_path):
                continue

            meta_path = os.path.join(match_path, 'meta.json')
            players_path = os.path.join(match_path, 'players.json')
            telemetry_path = os.path.join(match_path, 'telemetry.json')
            roster_path = os.path.join(match_path, 'rosters.json')  # roster.json 경로 추가

            meta_data = load_json(meta_path)
            players_data = load_json(players_path)
            telemetry_data = load_json(telemetry_path)
            roster_data = load_json(roster_path)  # roster.json 로드

            if not meta_data or not players_data or not telemetry_data or not roster_data:
                print(f"Missing data for match {match_id}.")
                continue

            # 팀 정보 추출
            team_info = extract_team_info(roster_data)

            # 매치 시작 시간 추출 (텔레메트리 이벤트 중 가장 이른 '_D'를 매치 시작 시간으로 간주)
            match_start_time = None
            for event in telemetry_data:
                event_time = parse_iso8601(event.get('_D', None))
                if event_time:
                    match_start_time = event_time
                    break
            if not match_start_time:
                print(f"Could not determine match start time for match {match_id}.")
                continue

            # 모든 플레이어 처리
            all_players = players_data

            for player in all_players:
                players_stats = player.get('attributes', {}).get('stats', {})
                account_id = players_stats.get('playerId', None)
                if not account_id:
                    continue

                # 플레이어의 teamId를 telemetry.json에서 추출
                # 해당 플레이어의 최초 LogPlayerPosition 이벤트에서 teamId를 추출
                player_team_id = None
                for event in telemetry_data:
                    if event.get('character', {}).get('accountId') == account_id:
                        player_team_id = event.get('character', {}).get('teamId', None)
                        if player_team_id is not None:
                            break
                if player_team_id is None:
                    team_details = {'team_rank': "None", 'team_won': "false"}
                else:
                    team_details = team_info.get(player_team_id, {'team_rank': "None", 'team_won': "false"})

                # 아이템 사용 정보 추출
                item_usage = extract_item_usage(telemetry_data, account_id)

                # 이동 경로 추출
                movement_routes = extract_movement_routes(telemetry_data, account_id, match_start_time)

                # 추가 데이터 추출
                additional_data = extract_additional_data(telemetry_data, account_id, players_stats)

                # 이동 경로 문자열로 변환
                movement_routes_str = ' -> '.join([f"({x},{y},{z})" for _, x, y, z in movement_routes]) if movement_routes else "None"

                # 첫 위치와 마지막 위치 추출
                first_location = movement_routes[0] if movement_routes else (None, None, None, None)
                last_location = movement_routes[-1] if movement_routes else (None, None, None, None)

                # elapsedTime 및 numAlivePlayers 추출 (예시: 첫 LogPlayerPosition 이벤트)
                elapsed_time = None
                num_alive_players = None
                for event in telemetry_data:
                    if event.get('_T') == 'LogPlayerPosition' and event.get('character', {}).get('accountId') == account_id:
                        elapsed_time = event.get('elapsedTime', None)
                        num_alive_players = event.get('numAlivePlayers', None)
                        break

                player_data.append({
                    'match_id': match_id,
                    'map_name': meta_data.get('mapName', None),
                    'game_mode': meta_data.get('gameMode', None),
                    'player_id': player.get('id', None),
                    'player_name': players_stats.get('name', None),
                    'player_account_id': account_id,
                    **item_usage,
                    **additional_data,
                    'movement_routes': movement_routes_str,
                    'first_location_x': first_location[1] if first_location[1] is not None else "None",
                    'first_location_y': first_location[2] if first_location[2] is not None else "None",
                    'first_location_z': first_location[3] if first_location[3] is not None else "None",
                    'final_location_x': last_location[1] if last_location[1] is not None else "None",
                    'final_location_y': last_location[2] if last_location[2] is not None else "None",
                    'final_location_z': last_location[3] if last_location[3] is not None else "None",
                    'walk_distance': players_stats.get('walkDistance', 0),
                    'swim_distance': players_stats.get('swimDistance', 0),
                    'ride_distance': players_stats.get('rideDistance', 0),
                    'road_kills': players_stats.get('roadKills', 0),
                    'vehicle_destroys': players_stats.get('vehicleDestroys', 0),
                    'weapons_acquired': players_stats.get('weaponsAcquired', 0),
                    'boosts': players_stats.get('boosts', 0),
                    'heals': players_stats.get('heals', 0),
                    'kill_streaks': players_stats.get('killStreaks', 0),
                    'headshot_kills': players_stats.get('headshotKills', 0),
                    'assists': players_stats.get('assists', 0),
                    'revives': players_stats.get('revives', 0),
                    'team_kills': players_stats.get('teamKills', 0),
                    'win_place': players_stats.get('winPlace', None),
                    'team_id': player_team_id if player_team_id else "None",
                    'team_rank': team_details.get('team_rank', "None"),
                    'team_won': team_details.get('team_won', "false"),
                    'elapsedTime': elapsed_time,
                    'numAlivePlayers': num_alive_players
                })

    return pd.DataFrame(player_data)

def main():
    """
    메인 함수: 모든 플레이어 데이터를 추출하고 CSV로 저장.
    """
    df_player_data = compile_player_data(DATA_DIR)

    if df_player_data.empty:
        print("No player data available.")
    else:
        output_csv = os.path.join(OUTPUT_DIR, 'player_data.csv')
        df_player_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Player data saved to '{output_csv}'.")
        print(df_player_data.head())

if __name__ == "__main__":
    main()
