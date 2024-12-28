import os
import json
import logging
from datetime import datetime, timezone
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from dateutil.parser import parse as parse_iso8601
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st
import time
import openai
import cv2

try:
    from chicken_dinner.pubgapi import PUBG
except ImportError:
    st.error("`chicken_dinner` 라이브러리가 설치되지 않았습니다. `pip install chicken-dinner` 명령어로 설치해주세요.")
    st.stop()

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# .env 파일 로드
env_path = os.path.join(os.getcwd(), '.env')
if not os.path.isfile(env_path):
    logging.error(f".env 파일을 찾을 수 없습니다: {env_path}")
    st.error(f".env 파일을 찾을 수 없습니다: {env_path}")
    st.stop()

load_dotenv(env_path)

PUBG_API_KEY = os.getenv("PUBG_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PUBG_API_KEY:
    logging.error("PUBG_API_KEY가 설정되지 않았습니다.")
    st.error("PUBG_API_KEY가 설정되지 않았습니다.")
    st.stop()

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

shard = "kakao"
pubg = PUBG(PUBG_API_KEY, shard=shard)
openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="PUBG Match Analysis Tool", layout="centered")

# CSS를 통해 좌우 여백 확보
# st.markdown("""
#     <style>
#     .block-container {
#         max-width: 80%;
#         margin: auto;
#         padding-left: 2rem;
#         padding-right: 2rem;
#     }
#     </style>
# """, unsafe_allow_html=True)

# 사이드바 메뉴: REPORT, GPTI 추가
st.sidebar.title("Menu")
menu = st.sidebar.radio("카테고리 선택", ["REPORT", "GPTI"])

st.title("PUBG Match Analysis Tool")
st.markdown("""이 애플리케이션은 PUBG 플레이어의 매치 데이터를 분석하고, 우승자와 비교한 결과를 제안합니다.""")

#################### GPTI 관련 함수 시작 ####################

def parse_iso8601_custom(timestamp_str):
    """ISO8601 형식의 타임스탬프 문자열을 datetime으로 변환하는 헬퍼 함수."""
    if timestamp_str is None:
        return None
    try:
        if '.' in timestamp_str:
            dot_index = timestamp_str.find('.')
            z_index = timestamp_str.find('Z', dot_index)
            if z_index == -1:
                z_index = len(timestamp_str)
            microseconds = timestamp_str[dot_index+1:z_index]
            if len(microseconds) > 6:
                microseconds = microseconds[:6]
            else:
                microseconds = microseconds.ljust(6, '0')
            timestamp_str = timestamp_str[:dot_index+1] + microseconds + timestamp_str[z_index:]
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

def load_json(file_path):
    """JSON 파일 로드"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"File Load Error: {file_path}\nError: {e}")
        return None

def calculate_distances(x_array, y_array):
    """Numba 없이 거리 계산."""
    distances = np.sqrt(x_array**2 + y_array**2)
    return distances

def extract_item_usage_gpti(telemetry_data, account_id):
    primary_weapon = None
    secondary_weapon = None
    armor_type = "None"
    health_items_used = 0
    boost_items_used = 0

    for event in telemetry_data:
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

def extract_movement_routes_gpti(telemetry_data, account_id, match_start_time):
    movement_routes = []
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
        timestamp_dt = parse_iso8601_custom(timestamp_str)
        if timestamp_dt and match_start_time:
            relative_seconds = (timestamp_dt - match_start_time).total_seconds()
            if x is not None and y is not None and z is not None:
                movement_routes.append((relative_seconds, x, y, z))

    movement_routes.sort(key=lambda x: x[0])
    return movement_routes

def extract_additional_data_gpti(telemetry_data, account_id, players_stats):
    items_carried = []
    loot_events = []
    combat_events = []

    for event in telemetry_data:
        if event.get('_T') == 'LogItemPickup' and event.get('character', {}).get('accountId') == account_id:
            item = event.get('item', {}).get('itemId', None)
            if item:
                items_carried.append(item)
                loot_events.append(event.get('_D', None))

        if event.get('_T') == 'LogPlayerAttack' and event.get('attacker', {}).get('accountId') == account_id:
            combat_events.append(event.get('_D', None))

    time_spent_looting = 0
    if loot_events:
        loot_timestamps = [parse_iso8601_custom(ts) for ts in loot_events if ts]
        loot_timestamps = [ts for ts in loot_timestamps if ts is not None]
        if loot_timestamps and len(loot_timestamps) > 1:
            time_spent_looting = (max(loot_timestamps) - min(loot_timestamps)).total_seconds()

    time_spent_in_combat = 0
    if combat_events:
        combat_timestamps = [parse_iso8601_custom(ts) for ts in combat_events if ts]
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

def extract_team_info_gpti(roster_data):
    team_info = {}
    for roster in roster_data:
        if roster.get('type') != 'roster':
            continue
        attributes = roster.get('attributes', {})
        stats = attributes.get('stats', {})
        rank = stats.get('rank', None)
        team_id = stats.get('teamId', None)
        won = attributes.get('won', "false")
        team_info[team_id] = {
            'team_rank': rank,
            'team_won': won
        }
    return team_info

def generate_single_csv_line(data_dir, user_name, match_id):
    """특정 user_name(플레이어 이름)과 match_id에 대해 JSON 데이터를 파싱하여 1행 CSV 만들기."""
    match_path = os.path.join(data_dir, user_name, match_id)
    if not os.path.exists(match_path):
        # 데이터 폴더 없으면 에러 대신 경고
        return None, f"데이터 폴더를 찾을 수 없습니다: {match_path}"

    meta_path = os.path.join(match_path, 'meta.json')
    players_path = os.path.join(match_path, 'players.json')
    telemetry_path = os.path.join(match_path, 'telemetry.json')
    roster_path = os.path.join(match_path, 'rosters.json')

    meta_data = load_json(meta_path)
    players_data = load_json(players_path)
    telemetry_data = load_json(telemetry_path)
    roster_data = load_json(roster_path)

    if not meta_data or not players_data or not telemetry_data or not roster_data:
        return None, "필요한 JSON 파일 중 일부를 로드할 수 없습니다."

    team_info = extract_team_info_gpti(roster_data)

    # 매치 시작 시간 결정
    match_start_time = None
    for event in telemetry_data:
        event_time = parse_iso8601_custom(event.get('_D', None))
        if event_time:
            match_start_time = event_time
            break
    if not match_start_time:
        return None, "매치 시작 시간을 결정할 수 없습니다."

    # 여기서 user_name으로 players.json에서 해당 플레이어 찾기
    # name 필드를 소문자로 비교
    target_account_id = None
    target_player_data = None
    for player in players_data:
        stats = player.get('attributes', {}).get('stats', {})
        p_name = stats.get('name', None)
        if p_name and p_name.lower() == user_name.lower():
            target_account_id = stats.get('playerId', None)  # accountId를 추출
            target_player_data = player
            break

    if not target_account_id or not target_player_data:
        return None, f"해당 user_name({user_name})에 해당하는 player를 찾을 수 없습니다."

    players_stats = target_player_data.get('attributes', {}).get('stats', {})

    # teamId 추출
    player_team_id = None
    for event in telemetry_data:
        if event.get('character', {}).get('accountId') == target_account_id:
            player_team_id = event.get('character', {}).get('teamId', None)
            if player_team_id is not None:
                break
    if player_team_id is None:
        team_details = {'team_rank': "None", 'team_won': "false"}
    else:
        team_details = team_info.get(player_team_id, {'team_rank': "None", 'team_won': "false"})

    # 아이템 사용 정보
    item_usage = extract_item_usage_gpti(telemetry_data, target_account_id)
    # 이동 경로
    movement_routes = extract_movement_routes_gpti(telemetry_data, target_account_id, match_start_time)
    # 추가 정보
    additional_data = extract_additional_data_gpti(telemetry_data, target_account_id, players_stats)

    movement_routes_str = ' -> '.join([f"({x:.1f},{y:.1f},{z:.1f})" for _, x, y, z in movement_routes]) if movement_routes else "None"

    first_location = movement_routes[0] if movement_routes else (None, None, None, None)
    last_location = movement_routes[-1] if movement_routes else (None, None, None, None)

    # elapsedTime 및 numAlivePlayers 추출
    elapsed_time = None
    num_alive_players = None
    for event in telemetry_data:
        if event.get('_T') == 'LogPlayerPosition' and event.get('character', {}).get('accountId') == target_account_id:
            elapsed_time = event.get('elapsedTime', None)
            num_alive_players = event.get('numAlivePlayers', None)
            break

    row_data = {
        'match_id': match_id,
        'map_name': meta_data.get('mapName', None),
        'game_mode': meta_data.get('gameMode', None),
        'player_id': target_player_data.get('id', None),
        'player_name': players_stats.get('name', None),
        'player_account_id': target_account_id,
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
    }

    OUTPUT_DIR = 'output_gpti'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, f'{user_name}_{match_id}_player_data.csv')

    if os.path.exists(output_csv):
        return None, "이미 해당 user_name과 match_id에 대한 CSV가 존재합니다."

    df = pd.DataFrame([row_data])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    return df, f"CSV 생성 완료: {output_csv}"

#################### GPTI 관련 함수 종료 ####################

if menu == "GPTI":
    # GPTI 메뉴 로직
    st.markdown("### GPTI 기능")
    st.markdown("user_id는 플레이어 이름이며, match_id를 입력하면 JSON에서 유저 정보를 추출하고 1행 CSV를 만듭니다.")

    data_dir = st.text_input("데이터 디렉토리 경로를 입력하세요:", "PUBG_data")
    user_name_input = st.text_input("user_name(플레이어 이름)을 입력하세요:")
    match_id_input = st.text_input("match_id를 입력하세요:")

    if st.button("JSON 정보 추출 및 CSV 생성"):
        if not user_name_input or not match_id_input:
            st.warning("user_name과 match_id를 모두 입력해주세요.")
        else:
            with st.spinner("데이터 처리 중..."):
                df_result, message = generate_single_csv_line(data_dir, user_name_input, match_id_input)
            if df_result is not None:
                st.success(message)
                st.dataframe(df_result)
            else:
                st.error(message)

    st.stop()

if 'match_ids' not in st.session_state:
    st.session_state.match_ids = []
if 'match_data' not in st.session_state:
    st.session_state.match_data = pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_player_matches(player_name):
    if not player_name:
        return []
    encoded_player_name = requests.utils.quote(player_name)
    url = f"https://api.pubg.com/shards/{shard}/players?filter[playerNames]={encoded_player_name}"
    headers = {
        'accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {PUBG_API_KEY}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        league_json = response.json()
        player_data = league_json.get('data', [])
        if not player_data:
            return []
        matches_data = player_data[0].get('relationships', {}).get('matches', {}).get('data', [])
        match_ids = [m['id'] for m in matches_data]
        return match_ids
    else:
        st.error(f"Player match request failed with status code {response.status_code}")
        return []

@st.cache_data(show_spinner=False)
def get_match_info(matchId):
    url = f'https://api.pubg.com/shards/{shard}/matches/{matchId}'
    headers = {
        'accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {PUBG_API_KEY}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        meta_info = data.get("data", {}).get("attributes", {})
        start_time_str = meta_info.get("createdAt", "")
        map_name = meta_info.get("mapName", "Unknown_Main")

        if start_time_str:
            start_datetime = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        else:
            start_datetime = None

        rosters = [item for item in data.get("included", []) if item.get("type") == "roster"]
        rosters_data = []
        for roster in rosters:
            participants = [p.get("id") for p in roster.get("relationships", {}).get("participants", {}).get("data", [])]
            rosters_data.append({
                "team_id": roster.get("id", ""),
                "participants": participants
            })

        participants = [item for item in data.get("included", []) if item.get("type") == "participant"]
        participant_id_to_name = {}
        participant_id_to_winplace = {}
        participant_id_to_kda = {}
        for participant in participants:
            participant_id = participant.get("id", "")
            attributes = participant.get("attributes", {})
            stats = attributes.get('stats', {})
            player_name = stats.get('name', "")
            win_place = stats.get('winPlace', None)
            kills = stats.get('kills', 0)
            assists = stats.get('assists', 0)
            death_type = stats.get('deathType', 'alive')
            deaths = 1 if death_type != 'alive' else 0

            participant_id_to_name[participant_id] = player_name
            participant_id_to_winplace[participant_id] = win_place
            participant_id_to_kda[participant_id] = {
                "kills": kills,
                "assists": assists,
                "deaths": deaths
            }

        match_folder = "matches"
        os.makedirs(match_folder, exist_ok=True)
        players = [item for item in data.get("included", []) if item.get("type") == "participant"]
        with open(os.path.join(match_folder, f"{matchId}_players.json"), "w", encoding="utf-8") as f:
            json.dump(players, f, ensure_ascii=False, indent=4)

        return start_datetime, rosters_data, participant_id_to_name, participant_id_to_winplace, participant_id_to_kda, map_name
    else:
        st.error(f"매치 {matchId} 요청 실패: {response.status_code}")
        return None, [], {}, {}, {}, "Unknown_Main"

@st.cache_data(show_spinner=False)
def get_telemetry_data(matchId):
    try:
        current_match = pubg.match(matchId)
        telemetry = current_match.get_telemetry()
        if telemetry:
            return telemetry
        else:
            st.error(f"텔레메트리 데이터 없음: {matchId}")
            return None
    except Exception as e:
        st.error(f"텔레메트리 가져오기 오류: {e}")
        return None

def is_match_ended(telemetry):
    for event in telemetry.events:
        event_type = getattr(event, "_T", "")
        if event_type == "LogMatchEnd":
            return True
    return False

def extract_rank(rosters, participant_id_to_winplace):
    rank_dict = {}
    for roster in rosters:
        if isinstance(roster, dict):
            team_id = roster.get('team_id', '')
            participant_ids = roster.get('participants', [])
            team_win_places = [participant_id_to_winplace.get(pid, None) for pid in participant_ids]
            team_win_places = [wp for wp in team_win_places if wp is not None]
            if team_win_places:
                team_win_place = min(team_win_places)
                rank_dict[team_id] = team_win_place
    return rank_dict

def extract_player_positions(telemetry, player_name, match_start_time):
    positions = []
    for event in telemetry.events:
        event_type = getattr(event, "_T", "")
        if event_type == "LogPlayerPosition":
            character = getattr(event, "character", None)
            if character:
                name = getattr(character, "name", "")
                if name.lower() == player_name.lower():
                    location = getattr(character, "location", {})
                    x = getattr(location, "x", 0.0)
                    y = getattr(location, "y", 0.0)
                    z = getattr(location, "z", 0.0)
                    timestamp_str = getattr(event, "_D", "")
                    if timestamp_str:
                        event_time = parse_iso8601(timestamp_str)
                        relative_seconds = (event_time - match_start_time).total_seconds()
                        if relative_seconds < 0:
                            continue
                    else:
                        relative_seconds = 0.0
                    positions.append((relative_seconds, x, y, z))
    return positions

def add_text_to_image(image, text, position, font_path='C:\\Windows\\Fonts\\malgun.ttf', font_size=16, color=(255, 255, 255)):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)

def visualize_movement_routes(movement_routes_dict, map_image_path, map_name, map_extents, zones, participant_id_to_name, fig_size=8):
    """
    Matplotlib의 imshow와 plot을 사용한 이동 경로 시각화
    """
    if not movement_routes_dict:
        st.error("이동 경로 데이터가 없습니다.")
        return

    if map_name not in map_extents:
        st.error(f"맵 extents가 정의되지 않았습니다: {map_name}")
        return

    # 맵 범위 설정
    min_x, max_x, min_y, max_y = map_extents[map_name]

    # 맵 이미지 로드
    if not os.path.isfile(map_image_path):
        st.error(f"맵 이미지 로드 실패: {map_image_path}")
        return

    img = plt.imread(map_image_path)

    # matplotlib 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 필요시
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # 맵 이미지 표시 (extent 설정)
    ax.imshow(img, extent=(min_x, max_x, max_y, min_y), origin='upper')

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    color_idx = 0

    # 각 플레이어별 이동 경로 plot
    for pid, routes in movement_routes_dict.items():
        if not routes:
            continue
        player_color = colors[color_idx % len(colors)]
        color_idx += 1

        player_name = participant_id_to_name.get(pid, "Unknown")

        # routes: (time, x, y, z)
        xs = [r[1] for r in routes]
        ys = [r[2] for r in routes]

        # 이동 경로를 선으로 연결
        ax.plot(xs, ys, marker='o', color=player_color, linestyle='-', linewidth=2, markersize=4, label=player_name)

        # 시작점 표시 (빨간색 원)
        ax.plot(xs[0], ys[0], 'o', color='red', markersize=10, label=f"{player_name} Start")

        # 종료점 표시 (녹색 사각형)
        ax.plot(xs[-1], ys[-1], 's', color='green', markersize=10, label=f"{player_name} End")

    # 존(White/Blue Zone) 표시 부분 (옵션)
    # zones: {'white': [...], 'blue': [...]}
    for zone_type, circles_data in zones.items():
        for circle in circles_data:
            if len(circle) < 4:
                continue
            _, cx, cy, radius = circle
            from matplotlib.patches import Circle
            zone_color = 'white' if zone_type == 'white' else 'blue'
            circle_patch = Circle((cx, cy), radius, fill=False, edgecolor=zone_color, linewidth=2)
            ax.add_patch(circle_patch)

    ax.set_title(f"{map_name} 맵에서의 플레이어 이동 경로")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()

    st.pyplot(fig, clear_figure=True)


def generate_replay(user_id, match_id, output_filename="recent_match.html"):
    try:
        with st.spinner("리플레이 생성 중... (약 1분 소요)"):
            user = pubg.players_from_names(user_id)[0]

            if match_id not in user.match_ids:
                st.warning(f"'{user_id}' 플레이어의 매치 목록에 {match_id}가 없습니다.")
                return

            match = pubg.match(match_id)
            telemetry = match.get_telemetry()
            try:
                telemetry.playback_animation(output_filename)
            except Exception as e:
                st.error(f"리플레이 생성 중 오류 발생: {e}")
                return

            st.success(f"리플레이 생성 완료: {output_filename}")

            with open(output_filename, "rb") as f:
                file_data = f.read()
            st.download_button(
                label="리플레이 HTML 다운로드",
                data=file_data,
                file_name=output_filename,
                mime='text/html'
            )
    except Exception as e:
        st.error(f"오류 발생: {e}")

def main_streamlit_app():
    player_name_input = st.text_input("플레이어 이름을 입력하세요:")
    if st.button("플레이어 매치 가져오기"):
        if not player_name_input:
            st.warning("플레이어 이름을 입력해주세요.")
            return

        with st.spinner("매치 데이터 가져오는 중..."):
            match_ids = get_player_matches(player_name_input)
            st.session_state.match_ids = match_ids

        if not st.session_state.match_ids:
            st.warning(f"플레이어 '{player_name_input}'의 매치를 찾을 수 없습니다.")
            return

        total_matches = len(st.session_state.match_ids)
        if total_matches > 0:
            st.write(f"매치 정보 수집 중... 총 {total_matches}개 매치 처리 필요 (매치당 약 1초 예상)")
            progress_bar = st.progress(0)
            match_info = []
            for i, match_id in enumerate(st.session_state.match_ids):
                start_time, rosters_data, participant_id_to_name, participant_id_to_winplace, participant_id_to_kda, map_name = get_match_info(match_id)
                if start_time:
                    match_info.append({
                        "match_id": match_id,
                        "start_time": start_time,
                        "rosters": rosters_data,
                        "participant_id_to_name": participant_id_to_name,
                        "participant_id_to_winplace": participant_id_to_winplace,
                        "participant_id_to_kda": participant_id_to_kda,
                        "map_name": map_name
                    })
                time.sleep(1)
                progress_bar.progress(int((i+1)/total_matches*100))

            match_df = pd.DataFrame(match_info)
            if not match_df.empty:
                match_df['start_time'] = pd.to_datetime(match_df['start_time'], utc=True)
                st.session_state.match_data = match_df.dropna(subset=["start_time"]).sort_values(by="start_time", ascending=False)
            else:
                st.session_state.match_data = pd.DataFrame()

            st.success("매치 정보 수집 완료!")

        if not st.session_state.match_data.empty:
            st.subheader("매치 목록")
            st.dataframe(st.session_state.match_data[['match_id', 'start_time']])

    if not st.session_state.match_data.empty:
        all_dates = st.session_state.match_data['start_time'].dt.date.unique()
        if len(all_dates) == 0:
            st.warning("매치 데이터가 없습니다.")
            return

        selected_date = st.date_input("날짜 선택:", value=all_dates.min(), min_value=all_dates.min(), max_value=all_dates.max())

        if selected_date not in all_dates:
            st.warning("유효한 날짜를 선택해주세요.")
            return

        filtered_matches = st.session_state.match_data[st.session_state.match_data['start_time'].dt.date == selected_date]

        if filtered_matches.empty:
            st.warning("선택한 날짜에 해당하는 매치가 없습니다.")
        else:
            st.subheader(f"{selected_date}의 매치 목록")
            st.dataframe(filtered_matches[['match_id', 'start_time']])

            selected_match_id = st.selectbox("해당 날짜의 매치를 선택하세요:", filtered_matches['match_id'].tolist())

            if selected_match_id:
                selected_match_row = filtered_matches[filtered_matches['match_id'] == selected_match_id].iloc[0]
                match_start_time = selected_match_row['start_time']

                with st.spinner("텔레메트리 데이터 가져오는 중..."):
                    telemetry = get_telemetry_data(selected_match_id)
                    if telemetry is None:
                        st.error("텔레메트리 데이터를 가져올 수 없습니다.")
                        return

                match_ended = is_match_ended(telemetry)
                st.write(f"매치 종료 여부: {match_ended}")

                if not match_ended:
                    st.warning("매치가 아직 종료되지 않았거나, 우승 팀을 찾을 수 없습니다.")
                    return

                match_row = st.session_state.match_data[st.session_state.match_data['match_id'] == selected_match_id].iloc[0]
                rosters = match_row['rosters']
                participant_id_to_name = match_row['participant_id_to_name']
                participant_id_to_kda = match_row['participant_id_to_kda']
                map_name = match_row['map_name']
                participant_id_to_winplace = match_row['participant_id_to_winplace']

                rank_dict = extract_rank(rosters, participant_id_to_winplace)
                winner_team_ids = [team_id for team_id, win_place in rank_dict.items() if win_place == 1]

                if not winner_team_ids:
                    st.warning("우승자 팀을 찾을 수 없습니다.")
                    winner_players = []
                else:
                    winner_players = []
                    for winner_team_id in winner_team_ids:
                        winner_roster = next((r for r in rosters if r.get('team_id') == winner_team_id), None)
                        if winner_roster:
                            winner_participant_ids = winner_roster.get('participants', [])
                            current_winners = [participant_id_to_name.get(pid, "") for pid in winner_participant_ids if participant_id_to_name.get(pid, "")]
                            winner_players.extend(current_winners)

                # 사용자와 우승자 이름 다른 줄에 표시
                st.write("**사용자 이름:**")
                st.write(player_name_input)
                st.write("**우승자(1등 팀) 플레이어:**")
                for w in winner_players:
                    st.write(w)

                def get_kda_info(pid_dict, pid):
                    if pid and pid in pid_dict:
                        kills = pid_dict[pid]['kills']
                        assists = pid_dict[pid]['assists']
                        deaths = pid_dict[pid]['deaths']
                        kda = (kills + assists) / max(deaths, 1)
                        return kills, assists, kda
                    else:
                        return np.nan, np.nan, np.nan

                player_name_lower = player_name_input.lower()
                user_participant_id = None
                for pid, pname in participant_id_to_name.items():
                    if pname.lower() == player_name_lower:
                        user_participant_id = pid
                        break

                user_kills, user_assists, user_kda = get_kda_info(participant_id_to_kda, user_participant_id)

                winner_kills_list = []
                winner_assists_list = []
                winner_kda_list = []
                for winner in winner_players:
                    winner_pid = None
                    for pid, pname in participant_id_to_name.items():
                        if pname.lower() == winner.lower():
                            winner_pid = pid
                            break
                    w_kills, w_assists, w_kda = get_kda_info(participant_id_to_kda, winner_pid)
                    if not np.isnan(w_kills):
                        winner_kills_list.append(w_kills)
                    if not np.isnan(w_assists):
                        winner_assists_list.append(w_assists)
                    if not np.isnan(w_kda):
                        winner_kda_list.append(w_kda)

                winner_kills = np.mean(winner_kills_list) if winner_kills_list else np.nan
                winner_assists = np.mean(winner_assists_list) if winner_assists_list else np.nan
                winner_kda = np.mean(winner_kda_list) if winner_kda_list else np.nan

                st.subheader("우승자와의 비교")
                if not np.isnan(user_kda) and not np.isnan(winner_kda) and winner_kda > 0:
                    def safe_val(val):
                        return "No Data" if (isinstance(val, float) and np.isnan(val)) else f"{val:.2f}"

                    comparison_df = pd.DataFrame({
                        "Metric": ["Kills", "Assists", "KDA"],
                        "Your Score": [safe_val(user_kills), safe_val(user_assists), safe_val(user_kda)],
                        "Winner's Score": [safe_val(winner_kills), safe_val(winner_assists), safe_val(winner_kda)]
                    })

                    st.table(comparison_df)
                else:
                    st.warning("사용자 또는 우승자의 데이터가 부족하여 비교가 어렵습니다.")

                st.subheader("플레이어 이동 경로 시각화")
                if not np.isnan(user_kda) and not np.isnan(winner_kda) and winner_kda > 0:
                    user_positions = extract_player_positions(telemetry, player_name_input, match_start_time)
                    if user_positions:
                        movement_routes = {}
                        user_pid = None
                        for pid, pname in participant_id_to_name.items():
                            if pname.lower() == player_name_lower:
                                user_pid = pid
                                break

                        if user_pid:
                            movement_routes[user_pid] = user_positions

                        for winner in winner_players:
                            w_pid = None
                            for pid, pname in participant_id_to_name.items():
                                if pname.lower() == winner.lower():
                                    w_pid = pid
                                    break
                            if w_pid:
                                w_positions = extract_player_positions(telemetry, winner, match_start_time)
                                movement_routes[w_pid] = w_positions

                        all_players = []
                        pid_to_pname = {}
                        for p_id in movement_routes.keys():
                            pname = participant_id_to_name.get(p_id, "Unknown")
                            pid_to_pname[p_id] = pname
                            all_players.append(pname)

                        selected_players = st.multiselect("시각화할 플레이어를 선택하세요:", all_players, default=all_players)

                        if st.button("시각화 갱신"):
                            filtered_routes = {}
                            for p_id, rt in movement_routes.items():
                                pname = pid_to_pname[p_id]
                                if pname in selected_players:
                                    filtered_routes[p_id] = rt

                            map_background_images = {
                                "Baltic_Main": "Erangel_Main_Low_Res.png",
                                "Desert_Main": "Miramar_Main_Low_Res.png",
                                "Range_Main": "Camp_Jackal_Main_Low_Res.png",
                                "PillarCompound_Main": "Training_Main_Low_Res.png",
                                "Kiki_Main": "Deston_Main_Low_Res.png",
                                "Italy_TDM_Main": "Rondo_Main_Low_Res.png",
                                "DihorOtok_Main": "Vikendi_Main_Low_Res.png",
                                "Boardwalk_Main": "Haven_Main_Low_Res.png",
                                "Savage_Main": "Sanhok_Main_Low_Res.png",
                                "Tiger_Main": "Taego_Main_Low_Res.png",
                                "Neon_Main": "Karakin_Main_Low_Res.png",
                                "Summerland_Main": "Paramo_Main_Low_Res.png",
                                "Chimera_Main": "Chimera_Main_Low_Res.png",
                                "Heaven_Main": "Heaven_Main_Low_Res.png"
                            }

                            map_images_path = r"C:\Users\inho0\Downloads\DE30-final-3-main\DE30-final-3-main\ml\heatmap\map_images"
                            map_image_filename = os.path.join(map_images_path, map_background_images.get(map_name, 'default.png'))

                            if not os.path.isfile(map_image_filename):
                                st.error(f"맵 이미지 파일이 존재하지 않습니다: {map_image_filename}")
                                return


                            converted_routes = {}
                            for p, rt in filtered_routes.items():
                                conv = []
                                for r in rt:
                                    timestamp, x, y, z = r
                                    x_m = x / 100
                                    y_m = y / 100
                                    z_m = z / 100
                                    conv.append((timestamp, x_m, y_m, z_m))
                                converted_routes[p] = conv

                            zones = {}
                            visualize_movement_routes(
                                movement_routes_dict=converted_routes,
                                map_image_path=map_image_filename,
                                map_name=map_name,
                                map_extents={
                                    'Desert_Main': [0, 8160.00, 0, 8160.00],
                                    'Baltic_Main': [0, 8160.00, 0, 8160.00],
                                    'DihorOtok_Main': [0, 8160.00, 0, 8160.00],
                                    'Erangel_Main': [0, 8160.00, 0, 8160.00],
                                    'Tiger_Main': [0, 8160.00, 0, 8160.00],
                                    'Neon_Main': [0, 8160.00, 0, 8160.00],
                                    'Kiki_Main': [0, 8160.00, 0, 8160.00],
                                    'Savage_Main': [0, 4080.00, 0, 4080.00],
                                    'Chimera_Main': [0, 3060.00, 0, 3060.00],
                                    'Summerland_Main': [0, 2040.00, 0, 2040.00],
                                    'Heaven_Main': [0, 1020.00, 0, 1020.00]
                                },
                                zones=zones,
                                participant_id_to_name=pid_to_pname,
                            )

                st.subheader("개선점 제안 (OpenAI API 활용)")
                if not np.isnan(user_kda) and not np.isnan(winner_kda) and winner_kda > 0:
                    def metric_str(val):
                        return 'No Data' if (isinstance(val, float) and np.isnan(val)) else f"{val:.2f}"

                    prompt = f"""사용자의 PUBG 매치 데이터와 우승자의 데이터를 비교했습니다. - 사용자의 킬: {metric_str(user_kills)} - 우승자의 킬: {metric_str(winner_kills)} - 사용자의 어시스트: {metric_str(user_assists)} - 우승자의 어시스트: {metric_str(winner_assists)} - 사용자의 KDA: {metric_str(user_kda)} - 우승자의 KDA: {metric_str(winner_kda)}  이 데이터를 바탕으로 사용자가 게임에서 개선할 수 있는 점을 구체적으로 제안해 주세요. 문장을 올바르게 끝내주세요."""

                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            n=1,
                            stop=None,
                            temperature=0.7,
                        )
                        suggestion = response.choices[0].message['content'].strip()
                        st.write(suggestion)
                    except Exception as e:
                        st.error(f"OpenAI API 오류: {e}")
                else:
                    st.warning("개선점을 제안할 충분한 데이터가 없습니다.")

                if st.checkbox("디버깅용 DataFrame 보기"):
                    st.subheader("DataFrames Structure")
                    if not st.session_state.match_data.empty:
                        with st.expander("match_df"):
                            st.dataframe(st.session_state.match_data.head())

                if st.button("CSV로 데이터 저장"):
                    match_df = st.session_state.match_data
                    match_df.to_csv("match_data.csv", index=False)
                    st.success("CSV 저장 완료.")

                st.subheader("리플레이 HTML 생성 및 다운로드")
                if st.button("리플레이 생성"):
                    generate_replay(user_id=player_name_input, match_id=selected_match_id, output_filename="recent_match.html")


def main():
    main_streamlit_app()

if __name__ == "__main__":
    main()
