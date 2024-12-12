import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv
from numba import njit
import numpy as np

# .env 파일 경로 설정 및 로드
env_path = r"C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\.gitignore\.env"
load_dotenv(env_path)

# .env 파일에서 API 키 가져오기
API_KEY = os.getenv("PUBG_API_KEY")

@njit
def calculate_distances(x_array, y_array):
    """
    Numba를 활용하여 플레이어의 이동 거리 계산을 가속화.

    Args:
        x_array (numpy.ndarray): x 좌표 배열.
        y_array (numpy.ndarray): y 좌표 배열.

    Returns:
        numpy.ndarray: 각 위치에서의 거리.
    """
    distances = np.empty(x_array.shape[0], dtype=np.float64)
    for i in range(x_array.shape[0]):
        distances[i] = np.sqrt(x_array[i] ** 2 + y_array[i] ** 2)
    return distances

async def fetch_match_ids(session, player_name, api_key):
    """
    주어진 플레이어 이름으로 PUBG API에서 매치 ID 리스트를 가져옴

    Args:
        session (aiohttp.ClientSession): aiohttp 세션 객체.
        player_name (str): 플레이어의 이름.
        api_key (str): PUBG API 인증 키.

    Returns:
        list: 매치 ID 리스트.
    """
    url = f"https://api.pubg.com/shards/kakao/players?filter[playerNames]={player_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/vnd.api+json"
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                user_data = await response.json()
                # 플레이어의 매치 ID 추출
                matches = user_data.get("data", [{}])[0].get("relationships", {}).get("matches", {}).get("data", [])
                match_ids = [match.get("id") for match in matches]
                print(f"{player_name}의 match_ids: {match_ids}")
                return match_ids
            else:
                print(f"{player_name}의 match_ids를 가져오는 데 실패했습니다. 상태 코드: {response.status}")
                return []
    except Exception as e:
        print(f"{player_name}의 match_ids를 가져오는 중 오류 발생: {e}")
        return []

async def fetch_match_data_separately(session, match_id, player_name, api_key):
    """
    매치 데이터를 비동기적으로 가져와 match_id 별로 폴더를 만들어 관리.
    매치 메타 정보, 로스터 정보, 플레이어 정보, 텔레메트리 데이터를 저장.

    Args:
        session (aiohttp.ClientSession): aiohttp 세션 객체.
        match_id (str): 매치의 고유 ID.
        player_name (str): 플레이어의 이름.
        api_key (str): PUBG API 인증 키.
    """
    url = f"https://api.pubg.com/shards/kakao/matches/{match_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/vnd.api+json"
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                match_data = await response.json()
                # 저장 경로 생성: PUBG_data/{player_name}/{match_id}/ 구조
                match_folder = os.path.join("PUBG_data", player_name, match_id)
                os.makedirs(match_folder, exist_ok=True)

                # 매치 메타 정보 저장
                meta_info = match_data.get("data", {}).get("attributes", {})
                with open(os.path.join(match_folder, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta_info, f, ensure_ascii=False, indent=4)

                # 로스터(팀) 정보 저장
                rosters = [item for item in match_data.get("included", []) if item.get("type") == "roster"]
                with open(os.path.join(match_folder, "rosters.json"), "w", encoding="utf-8") as f:
                    json.dump(rosters, f, ensure_ascii=False, indent=4)

                # 플레이어 정보 저장
                players = [item for item in match_data.get("included", []) if item.get("type") == "participant"]
                with open(os.path.join(match_folder, "players.json"), "w", encoding="utf-8") as f:
                    json.dump(players, f, ensure_ascii=False, indent=4)

                print(f"{player_name} - 매치 {match_id} 데이터 저장 완료!")

                # Telemetry URL 추출 및 처리
                telemetry_url = None
                for asset in match_data.get("included", []):
                    if asset.get("type") == "asset":
                        telemetry_url = asset.get("attributes", {}).get("URL")
                        break

                if telemetry_url:
                    await fetch_telemetry_data(session, telemetry_url, match_id, player_name)
                else:
                    print(f"{player_name} - 매치 {match_id}의 Telemetry URL을 찾을 수 없습니다.")
            else:
                print(f"{player_name} - 매치 {match_id} 데이터를 가져오는 데 실패했습니다. 상태 코드: {response.status}")
    except Exception as e:
        print(f"{player_name} - 매치 {match_id} 데이터를 가져오는 중 오류 발생: {e}")

async def fetch_telemetry_data(session, telemetry_url, match_id, player_name):
    """
    텔레메트리 데이터를 비동기적으로 가져와 저장하고, 데이터를 처리하여 결과를 저장.
    또한, 생존 시간이 가장 길고 이동 거리가 가장 큰/작은 플레이어의 이름과 관련 정보를 기록.

    Args:
        session (aiohttp.ClientSession): aiohttp 세션 객체.
        telemetry_url (str): 텔레메트리 데이터의 URL.
        match_id (str): 매치의 고유 ID.
        player_name (str): 플레이어의 이름.
    """
    try:
        async with session.get(telemetry_url) as response:
            if response.status == 200:
                telemetry_data = await response.json()
                # 텔레메트리 데이터 저장 경로 설정
                telemetry_path = os.path.join("PUBG_data", player_name, match_id, "telemetry.json")
                with open(telemetry_path, "w", encoding="utf-8") as f:
                    json.dump(telemetry_data, f, ensure_ascii=False, indent=4)
                print(f"{player_name} - 매치 {match_id}의 Telemetry 데이터 저장 완료!")

                # 텔레메트리 데이터 처리 (예: 플레이어의 이동 거리 계산 및 생존 시간 기록)
                process_telemetry_data(telemetry_data, player_name, match_id)
            else:
                print(f"{player_name} - 매치 {match_id}의 Telemetry 데이터를 가져오는 데 실패했습니다. 상태 코드: {response.status}")
    except Exception as e:
        print(f"{player_name} - 매치 {match_id}의 Telemetry 데이터를 가져오는 중 오류 발생: {e}")

def process_telemetry_data(telemetry_data, player_name, match_id):
    """
    텔레메트리 데이터를 처리하여 플레이어의 이동 거리를 계산하고,
    생존 시간이 가장 길고 이동 거리가 가장 큰/작은 플레이어의 모든 관련 정보를 기록.

    Args:
        telemetry_data (dict 또는 list): 텔레메트리 데이터.
        player_name (str): 플레이어의 이름.
        match_id (str): 매치의 고유 ID.
    """
    try:
        # telemetry_data가 dict인지 list인지 확인
        if isinstance(telemetry_data, dict):
            movements = telemetry_data.get("data", [])
        elif isinstance(telemetry_data, list):
            movements = telemetry_data
        else:
            movements = []

        # 모든 플레이어의 생존 시간과 이동 거리를 기록할 딕셔너리 초기화
        player_stats = {}

        # 플레이어의 이동 경로 및 생존 시간 추출
        for event in movements:
            if not isinstance(event, dict):
                continue  # 이벤트가 dict가 아니면 건너뜀
            event_type = event.get("_T")
            if event_type == "LogPlayerPosition":
                character = event.get("character", {})
                if not isinstance(character, dict):
                    continue  # character가 dict가 아니면 건너뜀
                player_id = character.get("accountId")
                if not player_id:
                    continue
                location = character.get("location", {})
                if not isinstance(location, dict):
                    continue  # location이 dict가 아니면 건너뜀
                x = location.get("x", 0.0)
                y = location.get("y", 0.0)
                timestamp = event.get("timestamp", 0.0)

                if player_id not in player_stats:
                    player_stats[player_id] = {
                        "positions": [],
                        "last_timestamp": 0.0
                    }
                player_stats[player_id]["positions"].append((x, y, timestamp))
                # 업데이트된 생존 시간 기록
                if timestamp > player_stats[player_id]["last_timestamp"]:
                    player_stats[player_id]["last_timestamp"] = timestamp

        # 각 플레이어의 총 이동 거리 계산
        player_distances = {}
        for player_id, stats in player_stats.items():
            positions = stats["positions"]
            if len(positions) < 2:
                player_distances[player_id] = 0.0
                continue
            x_coords = np.array([pos[0] for pos in positions], dtype=np.float64)
            y_coords = np.array([pos[1] for pos in positions], dtype=np.float64)
            distances = calculate_distances(x_coords, y_coords)
            total_distance = np.sum(distances)
            player_distances[player_id] = total_distance

        # 생존 시간이 가장 긴 플레이어 찾기
        if not player_stats:
            print(f"{player_name} - 매치 {match_id}에서 생존 시간이 가장 긴 플레이어를 찾을 수 없습니다.")
            return

        max_survival_time = max([stats["last_timestamp"] for stats in player_stats.values()])
        longest_survivors = [pid for pid, stats in player_stats.items() if stats["last_timestamp"] == max_survival_time]

        if not longest_survivors:
            print(f"{player_name} - 매치 {match_id}에서 생존 시간이 가장 긴 플레이어를 찾을 수 없습니다.")
            return

        # 생존 시간이 가장 긴 플레이어들 중 이동 거리가 가장 큰 플레이어 찾기
        longest_survivor_max_distance = max([player_distances[pid] for pid in longest_survivors])
        survivor_max_distance_ids = [pid for pid in longest_survivors if player_distances[pid] == longest_survivor_max_distance]

        # 생존 시간이 가장 긴 플레이어들 중 이동 거리가 가장 작은 플레이어 찾기
        longest_survivor_min_distance = min([player_distances[pid] for pid in longest_survivors])
        survivor_min_distance_ids = [pid for pid in longest_survivors if player_distances[pid] == longest_survivor_min_distance]

        # 플레이어 이름, walkDistance, timeSurvived 등의 모든 정보를 가져오기 위해 players.json 읽기
        players_json_path = os.path.join("PUBG_data", player_name, match_id, "players.json")
        if not os.path.exists(players_json_path):
            print(f"{player_name} - 매치 {match_id}의 players.json 파일이 존재하지 않습니다.")
            return

        with open(players_json_path, "r", encoding="utf-8") as f:
            players_data = json.load(f)

        # playerId를 key로 하는 딕셔너리 생성
        player_id_to_info = {}
        for player in players_data:
            player_attrs = player.get("attributes", {})
            stats = player_attrs.get("stats", {})
            player_id = stats.get("playerId")
            if player_id:
                player_id_to_info[player_id] = player  # 전체 플레이어 정보 저장

        # 생존 시간이 가장 길고 이동 거리가 가장 큰 플레이어 정보 수집
        longest_survivor_max_distance_info = []
        for pid in survivor_max_distance_ids:
            info = player_id_to_info.get(pid)
            if info:
                longest_survivor_max_distance_info.append(info)

        # 생존 시간이 가장 길고 이동 거리가 가장 작은 플레이어 정보 수집
        longest_survivor_min_distance_info = []
        for pid in survivor_min_distance_ids:
            info = player_id_to_info.get(pid)
            if info:
                longest_survivor_min_distance_info.append(info)

        # 결과 저장 경로 설정
        results_folder = os.path.join("PUBG_data", player_name, match_id, "results")
        os.makedirs(results_folder, exist_ok=True)

        # 결과를 하나의 JSON 파일에 저장
        results = {
            "longest_survivor_max_distance": longest_survivor_max_distance_info,
            "longest_survivor_min_distance": longest_survivor_min_distance_info
        }

        results_path = os.path.join(results_folder, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"{player_name} - 매치 {match_id}의 결과 정보 저장 완료!")

    except Exception as e:
        print(f"{player_name} - 매치 {match_id}의 텔레메트리 데이터 처리 중 오류 발생: {e}")

async def process_player(session, player_name, api_key):
    """
    주어진 플레이어의 모든 매치를 처리하는 함수.

    Args:
        session (aiohttp.ClientSession): aiohttp 세션 객체.
        player_name (str): 플레이어의 이름.
        api_key (str): PUBG API 인증 키.
    """
    print(f"=== {player_name} 처리 시작 ===")
    match_ids = await fetch_match_ids(session, player_name, api_key)
    if not match_ids:
        print(f"{player_name}에 대한 매치 ID가 없습니다.")
        return

    tasks = []
    for match_id in match_ids:
        task = asyncio.create_task(fetch_match_data_separately(session, match_id, player_name, api_key))
        tasks.append(task)

    # 현재 플레이어의 모든 매치 데이터를 비동기적으로 처리
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"=== {player_name} 처리 완료 ===\n")

async def main():
    """
    비동기 메인 함수: 각 플레이어를 순차적으로 처리하여 매치 데이터를 비동기적으로 가져와 처리.
    """
    player_names = [
        'EBBUNYA', 'SeulBing', 'PinGu_o0o', 'Xiwhawha4', 'car98-ace', 'Queen_Hyperion',
        'SLXPARIS', 'bot_0521', 'doringzzangg', 'BYC4_LYS', 'OverPath', 'Wooheesung-',
        'XI_hongxi', 'Apex_MounTaiN', 'SSENG-_-CHOCOBAR', 'EDGEBEAR', 'Ga_Bo_Ja_Gu',
        'Bang_YJH', 'AZA_Gangster', 'MERONG_OX', 'SLR_10Gak', 'GODMINJONG', 'doedoe_doe2',
        'Slbear-', 'J0KER___-', 'keehoons91', 'SEXY-_-HUN', 'Sulzzi_DR', 'Gangsi_da',
        'MERONG_XIN', 'D_-Kill', 'SAGICIJIMA', 'Chamber_-xxxXxxx', 'sulzzi_niss',
        'dododo_dodo_0', 'HORANG2464', 'vubyne68', 'ddudung-O_O7', 'jojaeho1111', 'JYM_97',
        '684_KYK', 'pizza_dou', 'KR_wound', 'I0S7F1P6', 'i___yoni', 'aaaaaaaaqdsws',
        'twich_win94', 'MERONG_X___X', 'JdG_78', 'AVEC_Kim_c_h', 'Zyossba', 'jhin0604',
        'C8NYEON-___-Lr7r', 'sayyoouhyun', 'asghjf', 'LOL-TAEJIN', 'backpause',
        'KR_ddooksim', 'ZERO__OPPA', 'ZERO__Kcal', '4Stack-to_ot', 'dotorimookim',
        'BYC4_OwO', 'junior_buugi', 'S_2xxn14', 'Teemo-_-x', 'lvA_mber', 'D_Lilghost',
        'AVEC_BBio', 
        'noljaninano', 'NotAfra1d', 'seoheelyn', 'AVEC_OK-Jin2', 'FM_HOiiiiii',
        'gurhhukjinn', 'Osan-188cm105kg', 'Back_-9', 'O_O-U_U-T_T', 'SSS_-MangGo', 'gasdhtw',
        'HERMES_HotGirl', 'Charlang_o3o', 'HERMES_meowpunch', 'THE_VSS', 'BABONYA', 'brave_ho',
        '13798285', 'Sage_-xxxxXxxxx', 'ImissApril25th', '174cm69kg', 'KR_48854885', '1P_TOT',
        'AVEC_SHILLA', 'C0lAP0LAR', 'nickdduck', 'VIRTUALGATE', 'MINCK0-0', 'BangSeungHak',
        'Dragon_Duck2', '1xk0vl'
    ]

    # Windows-specific event loop policy adjustment
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # aiohttp 세션을 사용하여 비동기적으로 API 호출
    timeout = aiohttp.ClientTimeout(total=60)  # Adjust timeout as needed
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for player_name in player_names:
            await process_player(session, player_name, API_KEY)

# 비동기 메인 함수 실행
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"RuntimeError 발생: {e}")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")
