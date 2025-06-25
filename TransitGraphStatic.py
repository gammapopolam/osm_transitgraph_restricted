import rustworkx as rx
import json
import shapely
from shapely.wkt import loads as wkt_loads
from scipy.spatial import KDTree
import numpy as np
from geopy.distance import geodesic
import geopandas as gpd
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt # Для базовых карт

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
from typing import List, Dict, Any, Optional, Union, Tuple
# --- Рефакторинг класса TransitGraph ---

with open('osm_transitgraph_restricted/url.md', 'r') as f:
    # Предполагается, что в файле url.md содержится URL для загрузки данных
    MB_URL = f.read().strip()



class TransitGraph:
    """
    Строит ориентированный граф для одного вида общественного транспорта с использованием rustworkx.
    Узлы представляют остановки, а ребра — сегменты между остановками в рамках маршрута.
    """
    def __init__(self,
                 trips_path: str,
                 stops_path: str,
                 s2s_path: str,
                 speed_kmh: float = 27, # Средняя скорость для этого вида транспорта в км/ч
                 transport_type: str = 'subway', # например, 'subway', 'bus', 'tram', 'commuter'
                 wc_mode: bool = False, # Учитывать доступность для инвалидных колясок при построении графа
                 keep_limited_wc: bool = True): # Сохранять остановки с 'limited' или 'cross-platform' доступность
        
        self.transport_type = transport_type
        self.speed_kmh = speed_kmh
        self.wc_mode = wc_mode
        self.keep_limited_wc = keep_limited_wc

        self.stops: List[Dict[str, Any]] = [] # Теперь список словарей
        self.s2s: List[Dict[str, Any]] = []    # Теперь список словарей
        self.trips: List[Dict[str, Any]] = [] # Структура данных Trips не полностью определена/используется в оригинале. Оставим как Dict[str, Any]

        self._load_data(trips_path, stops_path, s2s_path)
        self.graph = self._build_graph()

    def _load_data(self, trips_path: str, stops_path: str, s2s_path: str):
        """Загружает и парсит данные из JSON-файлов."""
        try:
            with open(stops_path, mode='r', encoding='utf-8') as f:
                raw_stops = json.loads(f.read())
                for s in raw_stops:
                    # Парсинг геометрии WKT один раз при загрузке и добавление в словарь
                    s['shapely_geom'] = wkt_loads(s['stop_shape'])
                    s['lat'] = s['shapely_geom'].y
                    s['lon'] = s['shapely_geom'].x
                    # Установить wheelchair для автобусов, если не указано
                    if self.transport_type == 'bus' and 'wheelchair' not in s:
                        s['wheelchair'] = 'yes'
                    elif 'wheelchair' not in s:
                         s['wheelchair'] = 'unknown' # По умолчанию 'unknown' для других типов
                    self.stops.append(s)
            # Создать быстрый поиск остановок по ID
            self._stop_id_map: Dict[str, Dict[str, Any]] = {str(stop['stop_id']): stop for stop in self.stops}

            with open(s2s_path, mode='r', encoding='utf-8') as f:
                raw_s2s = json.loads(f.read())
                for seg in raw_s2s:
                    # Парсинг геометрии WKT один раз при загрузке и добавление в словарь
                    seg['shapely_geom'] = wkt_loads(seg['shape'])
                    self.s2s.append(seg)

            with open(trips_path, mode='r', encoding='utf-8') as f:
                self.trips.extend(json.loads(f.read()))

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Требуемый файл данных не найден: {e.filename}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка декодирования JSON из файла: {e.msg} на строке {e.lineno}")
        except KeyError as e:
            raise ValueError(f"Отсутствует ожидаемый ключ в данных: {e}. Проверьте структуру входного JSON.")
        except Exception as e:
            raise RuntimeError(f"Произошла непредвиденная ошибка при загрузке данных: {e}")

    def _build_graph(self) -> rx.PyDiGraph:
        """Строит ориентированный граф rustworkx."""
        graph = rx.PyDiGraph()
        
        # Добавить узлы
        for stop in self.stops:
            # Применить фильтрацию доступности для инвалидных колясок, если wc_mode включен
            if self.wc_mode:
                access_level = self._get_accessibility_level(stop.get('wheelchair', 'unknown'))
                if access_level == 0: # Доступности 'no'
                    continue # Пропустить эту остановку
                # Если keep_limited_wc = False, также пропустить 'limited' или 'cross-platform' (access_level == 2)
                if not self.keep_limited_wc and access_level == 2:
                    continue
            
            node_idx = graph.add_node({
                'name': stop['stop_name'],
                'id': stop['stop_id'],
                'type': self.transport_type,
                'geom': stop['stop_shape'], # Оригинальная строка WKT
                'shapely_geom': stop['shapely_geom'], # Объект Shapely Point
                'wc_access': stop.get('wheelchair', 'unknown'),
                'lat': stop['lat'],
                'lon': stop['lon']
            })
            stop['node_idx'] = node_idx # Обновить словарь остановки с назначенным индексом узла

        # Добавить ребра
        for segment in self.s2s:
            from_stop = self._stop_id_map.get(str(segment['from']))
            to_stop = self._stop_id_map.get(str(segment['to']))

            if from_stop and to_stop and 'node_idx' in from_stop and 'node_idx' in to_stop:
                # Проверить, были ли остановки отфильтрованы из-за доступности (если wc_mode включен)
                if not graph.has_node(from_stop['node_idx']) or not graph.has_node(to_stop['node_idx']):
                    continue

                traveltime = self._calculate_travel_time(float(segment['length']), self.speed_kmh)
                edge_data = {
                    'type': self.transport_type,
                    'traveltime': traveltime,
                    'trip_id': segment['trip_id'],
                    'trip_ref': segment['trip_ref'],
                    'geom': segment['shapely_geom'], # Объект Shapely LineString
                    'length': float(segment['length']),
                    'from_node_idx': from_stop['node_idx'],
                    'to_node_idx': to_stop['node_idx'],
                    'leg_name': f"{from_stop['stop_name']}-{to_stop['stop_name']}"
                }
                graph.add_edge(from_stop['node_idx'], to_stop['node_idx'], edge_data)
            else:
                print(f"Предупреждение: Отсутствуют данные остановки для сегмента {segment['from']} -> {segment['to']}")

        print(f"Граф для {self.transport_type} построен с {graph.num_nodes()} узлами и {graph.num_edges()} ребрами.")
        return graph

    def _calculate_travel_time(self, length_meters: float, speed_kmh: float) -> float:
        """Рассчитывает время в пути в минутах."""
        if speed_kmh <= 0:
            return float('inf') # Избежать деления на ноль или неположительной скорости
        speed_mps = speed_kmh * 1000 / 3600 # Преобразовать км/ч в м/с
        traveltime_seconds = length_meters / speed_mps
        traveltime_minutes = round(traveltime_seconds / 60, 3)
        return traveltime_minutes
    
    def _get_accessibility_level(self, wheelchair_status: str) -> int:
        """
        Определяет уровень доступности: 0 (нет), 1 (да), 2 (ограниченная/пересадочная платформа).
        """
        if wheelchair_status == 'no':
            return 0
        elif wheelchair_status == 'yes':
            return 1
        elif wheelchair_status in ['cross-platform', 'limited']:
            return 2
        return 1 # По умолчанию доступно, если неизвестно/не указано


    def visualize_stops(self) -> gpd.GeoDataFrame:
        """
        Создает GeoDataFrame всех остановок, представленных в этом конкретном TransitGraph,
        для визуализации.
        """
        stops_data_for_gdf = []
        for stop in self.stops:
            # Включить только те остановки, которые фактически были добавлены в граф
            if 'node_idx' in stop and self.graph.has_node(stop['node_idx']):
                stops_data_for_gdf.append({
                    'node_idx': stop['node_idx'],
                    'stop_id': stop['stop_id'],
                    'stop_name': stop['stop_name'],
                    'type': self.transport_type, 
                    'wc_access': stop.get('wheelchair', 'unknown'),
                    'lat': stop['lat'],
                    'lon': stop['lon'],
                    'geometry': stop['shapely_geom'] # Использовать уже разобранную геометрию shapely
                })
        
        if not stops_data_for_gdf:
            print(f"Не найдено остановок для визуализации для {self.transport_type} графа.")
            return gpd.GeoDataFrame()

        stops_gdf = gpd.GeoDataFrame(stops_data_for_gdf)
        stops_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)
        return stops_gdf

    def visualize_edges(self) -> gpd.GeoDataFrame:
        """
        Создает GeoDataFrame всех транзитных сегментов (ребер) в этом конкретном TransitGraph,
        для визуализации.
        """
        edges_data_for_gdf = []
        # Изменение: Итерация по индексам ребер, а затем получение конечных точек и полезной нагрузки
        for edge_idx in self.graph.edge_indices():
            u, v = self.graph.get_edge_endpoints_by_index(edge_idx)
            edge_payload = self.graph.get_edge_data_by_index(edge_idx)
            
            # Если edge_payload не является словарем, то это неожиданный тип, пропускаем
            if not isinstance(edge_payload, dict):
                print(f"Предупреждение: Ребро {u}->{v} имеет неожиданный тип полезной нагрузки: {type(edge_payload)}. Пропуск визуализации.")
                continue

            # Включить только транзитные ребра (не пересадочные, так как это TransitGraph)
            # Тип ребра должен соответствовать типу транспорта TransitGraph
            if edge_payload.get('type') == self.transport_type and edge_payload.get('geom'):
                edges_data_for_gdf.append({
                    'type': edge_payload['type'],
                    'traveltime': edge_payload['traveltime'],
                    'from_node_idx': u, # Используем u, v, полученные из edge_endpoints
                    'to_node_idx': v,     # Используем u, v, полученные из edge_endpoints
                    'trip_id': edge_payload.get('trip_id'),
                    'ref': edge_payload.get('trip_ref'),
                    'length': edge_payload.get('length'),
                    'leg_name': edge_payload.get('leg_name'),
                    'geometry': edge_payload['geom'] # Использовать уже разобранную геометрию shapely
                })

        if not edges_data_for_gdf:
            print(f"Не найдено ребер для визуализации для {self.transport_type} графа.")
            return gpd.GeoDataFrame()

        edges_gdf = gpd.GeoDataFrame(edges_data_for_gdf)
        edges_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)
        return edges_gdf


# --- Рефакторинг класса EnhTransitGraph ---

class EnhTransitGraph:
    """
    Объединяет несколько транзитных графов и облегчает межмодальные пересадки.
    """
    def __init__(self,
                 transit_graphs: List[TransitGraph],
                 walking_speed_kmh: float = 3): # Скорость ходьбы для пересадок
        
        if not transit_graphs:
            raise ValueError("Должен быть предоставлен хотя бы один TransitGraph.")

        self.walking_speed_kmh = walking_speed_kmh
        self.graph: rx.PyDiGraph = self._combine_transit_graphs(transit_graphs)
        # self.pedestrian_graph удален

        # Построить KDTree для всех транзитных остановок в объединенном графе
        self._build_stop_kdtree()

        # _build_pedestrian_kdtree удален


    def _combine_transit_graphs(self, transit_graphs: List[TransitGraph]) -> rx.PyDiGraph:
        """Объединяет несколько транзитных графов rustworkx в один граф."""
        if len(transit_graphs) == 1:
            print("Предоставлен только один транзитный граф, используется напрямую.")
            return transit_graphs[0].graph
        else:
            combined_graph = rx.digraph_union(transit_graphs[0].graph, transit_graphs[1].graph)
            for i in range(2, len(transit_graphs)):
                print(f"Объединение графа {i+1}/{len(transit_graphs)}")
                combined_graph = rx.digraph_union(combined_graph, transit_graphs[i].graph)
            print(f"Объединенный граф имеет {combined_graph.num_nodes()} узлов и {combined_graph.num_edges()} ребер.")
            return combined_graph

    def _build_stop_kdtree(self):
        """Строит KDTree для всех транзитных остановок в объединенном графе."""
        stop_coords = [] # Список (лат, лон)
        self.stop_node_map: Dict[int, Any] = {} # Сопоставление node_idx с его данными для быстрого поиска

        for node_idx in self.graph.node_indices():
            node_data = self.graph[node_idx]
            # Исключить пешеходные узлы, если они существуют
            if node_data.get('type') != 'pedestrian': # Предполагаем, что пешеходных узлов больше нет, но оставляем проверку
                stop_coords.append((node_data['lat'], node_data['lon']))
                self.stop_node_map[node_idx] = node_data
        
        self.stop_node_indices_list = list(self.stop_node_map.keys()) # Упорядоченный список индексов узлов для KDTree
        self.stop_kdtree = KDTree(np.array(stop_coords))
        print(f"Построен KDTree для {len(self.stop_node_indices_list)} транзитных остановок.")

    # _build_pedestrian_kdtree удален

    @lru_cache(maxsize=None)
    def _get_kn_stops_node_idx(self, point: Tuple[float, float], k: int, distance_upper_bound_degrees: float) -> List[int]:
        """
        Находит k ближайших транзитных остановок к точке в пределах заданного расстояния в градусах.
        Возвращает список индексов узлов rustworkx.
        """
        distances, indices = self.stop_kdtree.query(point, k=k, distance_upper_bound=distance_upper_bound_degrees)
        
        # KDTree возвращает `inf` для расстояний за пределами `distance_upper_bound` и `len(array)` для индексов,
        # когда k больше, чем доступных точек, или точки находятся за пределами диапазона.
        # Отфильтровать эти неверные результаты.
        
        valid_node_indices = []
        # Убедиться, что 'indices' является массивом для последовательной итерации
        if isinstance(indices, (int, np.integer)): # Случай с одним результатом
            if distances != float('inf'):
                original_node_idx = self.stop_node_indices_list[indices]
                if self.graph.has_node(original_node_idx):
                    valid_node_indices.append(original_node_idx)
        else: # Случай с несколькими результатами
            for dist_val, idx_val in zip(distances, indices):
                if dist_val != float('inf') and idx_val < len(self.stop_node_indices_list):
                    original_node_idx = self.stop_node_indices_list[idx_val]
                    if self.graph.has_node(original_node_idx):
                        valid_node_indices.append(original_node_idx)
        
        return list(set(valid_node_indices)) # Использовать set для обеспечения уникальности, затем преобразовать обратно в список


    def init_interchanges(self, k: int = 4, search_radius_degrees: float = 0.009,
                          interchange_time_override: Optional[Union[float, str]] = 'distance_weighted',
                          max_interchange_distance_meters: float = 500):
        """
        Инициализирует межмодальные пересадочные ребра в объединенном транспортном графе.
        Он соединяет физически близко расположенные остановки, добавляя ребро 'interchange'
        с рассчитанным временем в пути.

        :param k: Количество ближайших остановок для рассмотрения для пересадки.
        :param search_radius_degrees: Начальный радиус пространственного поиска в градусах для KDTree.
                                      Это грубый фильтр; `max_interchange_distance_meters`
                                      обеспечивает окончательную проверку расстояния.
        :param interchange_time_override:
            - 'distance_weighted': Время в пути рассчитывается на основе геодезического расстояния и скорости ходьбы.
            - float: Фиксированное время в пути в минутах для всех пересадок.
        :param max_interchange_distance_meters: Максимальное геодезическое расстояние (в метрах)
                                                 между двумя остановками для создания пересадки.
        """
        print(f"Инициализация пересадок с k={k}, search_radius_degrees={search_radius_degrees}, "
              f"max_interchange_distance_meters={max_interchange_distance_meters}...")

        added_interchanges = 0
        for node_u_idx in self.graph.node_indices():
            node_u_data = self.graph[node_u_idx]
            # Проверки на 'pedestrian' тип узлов теперь не нужны, так как пешеходный граф удален
            # if node_u_data.get('type') == 'pedestrian':
            #     continue

            u_point = (node_u_data['lat'], node_u_data['lon'])
            
            # Использовать KDTree для начальной фильтрации по близости (в градусах)
            candidate_stop_indices = self._get_kn_stops_node_idx(u_point, k, search_radius_degrees)

            for node_v_idx in candidate_stop_indices:
                if node_u_idx == node_v_idx:
                    continue # Не добавлять пересадку с остановки на саму себя

                node_v_data = self.graph[node_v_idx]
                v_point = (node_v_data['lat'], node_v_data['lon'])

                # Рассчитать фактическое геодезическое расстояние в метрах
                distance_meters = geodesic(u_point, v_point).meters

                if distance_meters <= max_interchange_distance_meters:
                    traveltime = 0.0
                    if interchange_time_override == 'distance_weighted':
                        # Рассчитать время ходьбы + фиксированные накладные расходы на пересадку (например, 1 минута)
                        # Скорость в м/мин: self.walking_speed_kmh * 1000 / 60
                        if self.walking_speed_kmh > 0:
                            traveltime = (distance_meters / (self.walking_speed_kmh * 1000 / 60)) + 1 # 1 мин накладных расходов
                        else:
                            traveltime = float('inf')
                    elif isinstance(interchange_time_override, (int, float)):
                        traveltime = float(interchange_time_override)
                    else:
                        # По умолчанию или запасной вариант, если override недействителен
                        traveltime = (distance_meters / (self.walking_speed_kmh * 1000 / 60)) + 1 
                        
                    # Добавить ребро пересадки в обоих направлениях
                    # Проверить, существует ли ребро, чтобы избежать дубликатов
                    if not self.graph.has_edge(node_u_idx, node_v_idx):
                        self.graph.add_edge(node_u_idx, node_v_idx, {
                            'type': 'interchange',
                            'traveltime': traveltime,
                            'from_node_idx': node_u_idx,
                            'to_node_idx': node_v_idx
                        })
                        added_interchanges += 1
                        # print(f"Добавлена пересадка: {node_u_data['name']} ({node_u_data['type']}) "
                        #       f"-> {node_v_data['name']} ({node_v_data['type']}) "
                        #       f"время: {traveltime:.2f} мин, расстояние: {distance_meters:.0f}м")
        print(f"Завершено инициализация пересадок. Добавлено {added_interchanges} новых ребер.")


    # get_nearest_pedestrian_node_idx удален

    def visualize_stops(self) -> gpd.GeoDataFrame:
        """
        Создает GeoDataFrame всех транзитных остановок в объединенном графе для визуализации.
        """
        stations_data = []
        for node_idx in self.graph.node_indices():
            node_data = self.graph[node_idx]
            # Включить только транзитные остановки, а не пешеходные узлы или пересадочные ребра
            # (предполагаем, что пешеходных узлов больше нет)
            # if node_data.get('type') != 'pedestrian': 
            viz_data = node_data.copy()
            viz_data['node_idx'] = node_idx
            # Извлечь геометрию shapely_geom для использования в GeoDataFrame
            viz_data['geometry'] = viz_data.pop('shapely_geom') 
            # Удалить оригинальную строку WKT, если она больше не нужна
            viz_data.pop('geom', None) 
            stations_data.append(viz_data)
        
        if not stations_data:
            print("Не найдено транзитных остановок для визуализации.")
            return gpd.GeoDataFrame()

        stations_gdf = gpd.GeoDataFrame(stations_data)
        stations_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)
        return stations_gdf

    def visualize_edges(self) -> gpd.GeoDataFrame:
        """
        Создает GeoDataFrame всех транзитных сегментов (ребер) в этом конкретном TransitGraph,
        для визуализации.
        """
        edges_data = []
        # Изменение: Итерация по индексам ребер, а затем получение конечных точек и полезной нагрузки
        for edge_idx in self.graph.edge_indices():
            u, v = self.graph.get_edge_endpoints_by_index(edge_idx)
            edge_payload = self.graph.get_edge_data_by_index(edge_idx)
            
            # Если edge_payload не является словарем, то это неожиданный тип, пропускаем
            if not isinstance(edge_payload, dict):
                print(f"Предупреждение: Ребро {u}->{v} имеет неожиданный тип полезной нагрузки: {type(edge_payload)}. Пропуск визуализации.")
                continue

            # Включить только транзитные ребра (не пересадочные, так как это TransitGraph)
            # Тип ребра должен соответствовать типу транспорта TransitGraph
            if edge_payload.get('type') == self.transport_type and edge_payload.get('geom'):
                edges_data.append({
                    'type': edge_payload['type'],
                    'traveltime': edge_payload['traveltime'],
                    'from_node_idx': u, # Используем u, v, полученные из edge_endpoints
                    'to_node_idx': v,     # Используем u, v, полученные из edge_endpoints
                    'trip_id': edge_payload.get('trip_id'),
                    'ref': edge_payload.get('trip_ref'),
                    'length': edge_payload.get('length'),
                    'leg_name': edge_payload.get('leg_name'),
                    'geometry': edge_payload['geom'] # Использовать уже разобранную геометрию shapely
                })

        if not edges_data:
            print(f"Не найдено ребер для визуализации для {self.transport_type} графа.")
            return gpd.GeoDataFrame()

        edges_gdf = gpd.GeoDataFrame(edges_data)
        edges_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)
        return edges_gdf
import json
with open ('osm_transitgraph_restricted/colors.json', 'r') as f:
    # Предполагается, что в файле colors.json содержится словарь с цветами для трамвайных маршрутов
    DEFAULT_TRAM_COLORS = json.load(f)

class Operativka:
    """
    Класс для оперативного анализа транспортной сети, построенной TransitGraph.
    Позволяет получать информацию о затрагиваемых маршрутах, визуализировать их
    и выполнять фильтрацию.
    """
    def __init__(self, tg: TransitGraph, tram_colors_map: Optional[Dict[str, str]] = None,
                 remove_non_priority_refs: Optional[List[str]] = None):
        """
        Инициализирует класс Operativka.

        :param tg: Экземпляр TransitGraph, содержащий транспортный граф.
        :param tram_colors_map: Необязательный словарь для переопределения цветов трамвайных маршрутов.
                                Если None, используются цвета по умолчанию.
        :param remove_non_priority_refs: Необязательный список ссылок маршрутов ('ref'),
                                         которые должны быть удалены из графа при инициализации.
        """
        self.graph = tg.graph # Используем граф, построенный TransitGraph
        self.tram_colors = tram_colors_map if tram_colors_map is not None else DEFAULT_TRAM_COLORS
        
        # Если указаны маршруты для удаления, удаляем их
        if remove_non_priority_refs:
            self.remove_routes_by_refs(remove_non_priority_refs)

        # Trips dict должен быть получен после возможного удаления маршрутов
        self.trips_dict = self.get_trip_and_route_alignments()
        
        # Инициализация GeoDataFrame остановок
        self._initialize_stops_gdf()

    def _initialize_stops_gdf(self):
        """Инициализирует GeoDataFrame остановок из текущего графа."""
        stations_data = []
        for node_idx in self.graph.node_indices():
            data = self.graph[node_idx]
            # Убедиться, что 'shapely_geom' существует и является объектом shapely
            if 'shapely_geom' in data and isinstance(data['shapely_geom'], shapely.geometry.Point):
                data_copy = data.copy()
                data_copy['node_idx'] = node_idx
                data_copy['geometry'] = data_copy.pop('shapely_geom') # Переименовать для GeoDataFrame
                data_copy.pop('geom', None) # Удалить оригинальную WKT строку, если она хранилась
                stations_data.append(data_copy)
            else:
                print(f"Предупреждение: Узел {node_idx} отсутствует или имеет неверный формат 'shapely_geom'. Пропускается при инициализации stops_gdf.")

        self.stops_gdf = gpd.GeoDataFrame(stations_data)
        if not self.stops_gdf.empty:
            self.stops_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)
        else:
            print("Предупреждение: stops_gdf пуст после инициализации.")

    def remove_routes_by_refs(self, refs_to_remove: List[str]):
        """
        Удаляет все ребра из графа, относящиеся к указанным маршрутам (ref).
        
        :param refs_to_remove: Список строковых идентификаторов маршрутов ('ref'), которые нужно удалить.
        """


        #def remove_nonpriority_routes(graph):
    # def should_remove(edge):
    #     removing_list=['39а', '3а']
    #     return edge["ref"] in removing_list
    
    # edge_indices = graph.graph.edge_indices()
    # edges_to_remove = [
    #     edge_idx for edge_idx in edge_indices
    #     if should_remove(graph.graph.get_edge_data_by_index(edge_idx))
    # ]

    # # Remove the edges (need to reverse to maintain correct indices while removing)
    # for edge_idx in sorted(edges_to_remove, reverse=True):
    #     graph.graph.remove_edge_from_index(edge_idx)
    # return graph


        edges_to_remove_indices = []
        # Итерация по индексам ребер для безопасного удаления
        for edge_idx in self.graph.edge_indices():
            edge_data = self.graph.get_edge_data_by_index(edge_idx) # Получаем данные ребра
            
            if edge_data and edge_data.get('trip_ref') in refs_to_remove:
                edges_to_remove_indices.append(edge_idx)
        
        # Удаляем ребра в обратном порядке, чтобы избежать проблем со смещением индексов
        for edge_idx in sorted(edges_to_remove_indices, reverse=True):
            self.graph.remove_edge_from_index(edge_idx)
        
        print(f"Удалены ребра для маршрутов: {refs_to_remove}. Новое количество ребер: {self.graph.num_edges()}")
        # Пересоздаем trips_dict после удаления ребер
        self.trips_dict = self.get_trip_and_route_alignments()
    def get_affecting_routes(self, operative_osmids: Union[int, str, List[Union[int, str]]], override_order: Optional[List[str]] = None, 
                             remove_duplicates: bool = True, cutoff: int = 30) -> Dict[str, List[int]]:
        """
        Получает список остановок, затрагиваемых событием на указанных остановках.
        Для каждого trip_id возвращает ОДНУ непрерывную последовательность остановок,
        начинающуюся от одной из оперативных остановок и идущую назад по маршруту.
        Если trip_id затрагивается несколькими оперативными остановками, используется
        последовательность, найденная первой (по порядку operative_osmids).

        :param operative_osmids: OSM ID одной остановки (int/str) или список OSM ID остановок.
        :param remove_duplicates: Если True, удаляет частичные дубликаты из списков остановок маршрутов.
        :param cutoff: Максимальное время в пути (в минутах) для определения "затронутых" остановок.
        :return: Словарь, где ключ - trip_id, значение - список индексов узлов остановок.
        """
        if not isinstance(operative_osmids, list):
            operative_osmids = [operative_osmids]

        # This will store the *first found* stoplist for each trip_id
        final_stoplists_for_trips = {}
        affected_refs_all = set()

        for osmid in operative_osmids:
            operative_node_idx = self.get_node_idx_by_osmid(osmid)
            if operative_node_idx is None:
                print(f"Ошибка: Остановка с OSM ID {osmid} не найдена в графе. Пропускается.")
                continue

            # Get incoming edges to the current operative node
            incoming_edges = list(self.graph.in_edges(operative_node_idx))
            
            for u, v, edge_data in incoming_edges: # u is predecessor, v is current_node (operative_node_idx)
                trip_id = edge_data.get('trip_id')
                if trip_id and trip_id not in final_stoplists_for_trips: # Only process if this trip_id hasn't been added yet
                    # get_route_stoplist_from now returns correctly ordered list
                    stoplist_for_this_segment = self.get_route_stoplist_from(operative_node_idx, trip_id, cutoff=cutoff)
                    
                    if stoplist_for_this_segment: # Ensure we got a valid sequence
                        final_stoplists_for_trips[trip_id] = stoplist_for_this_segment
                        ref = self.get_ref_by_trip_id(trip_id)
                        if ref:
                            affected_refs_all.add(ref)

        print('Маршруты, затрагиваемые событием:', list(affected_refs_all))
        
        if remove_duplicates:
            if override_order is not None:
                final_stoplists_for_trips = self.remove_partial_duplicates(final_stoplists_for_trips, override_order)
            else:
                final_stoplists_for_trips = self.remove_partial_duplicates(final_stoplists_for_trips)
            
        return final_stoplists_for_trips

    def get_gdf_from_stoplist(self, stoplist: Dict[str, List[int]], save_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Создает GeoDataFrame из списка остановок, с добавлением информации о маршрутах и цветах.

        :param stoplist: Словарь, где ключ - trip_id, значение - список индексов узлов остановок.
        :param save_path: Путь для сохранения GeoJSON файла. Если None, файл не сохраняется.
        :return: GeoDataFrame с выбранными остановками и их атрибутами.
        """
        selections = []
        for trip_id, node_indices_list in stoplist.items():
            ref = self.get_ref_by_trip_id(trip_id)
            color = self.tram_colors.get(ref, self.tram_colors.get("Default", "#000000")) # Получаем цвет, с запасным вариантом
            
            # Фильтруем self.stops_gdf по node_idx и добавляем атрибуты
            selection = self.stops_gdf[self.stops_gdf['node_idx'].isin(node_indices_list)].copy()
            selection['trip_id'] = trip_id
            selection['ref'] = ref
            selection['color'] = color
            selections.append(selection)
        
        if not selections:
            print("Нет выбранных остановок для создания GeoDataFrame.")
            return gpd.GeoDataFrame()

        selections_gdf = gpd.GeoDataFrame(pd.concat(selections, ignore_index=True))
        if save_path:
            try:
                selections_gdf.to_file(save_path, driver='GeoJSON', encoding='utf-8')
                print(f"GeoDataFrame сохранен в {save_path}")
            except Exception as e:
                print(f"Ошибка при сохранении GeoJSON: {e}")
        return selections_gdf

    def remove_partial_duplicates(self, routes_dict: Dict[str, List[int]], override_order: List[str]) -> Dict[str, List[int]]:
        """
        Удаляет частичные дубликаты последовательностей остановок из словаря маршрутов,
        учитывая приоритет маршрутов из override_order.

        :param routes_dict: Словарь маршрутов, где ключ - trip_id, значение - список индексов узлов остановок.
        :param override_order: Список route_ref, которые должны быть обработаны первыми.
        :return: Словарь маршрутов без частичных дубликатов, с учетом приоритета.
        """
        # Создаем маппинг trip_id -> route_ref для быстрого доступа
        trip_to_route_map: Dict[str, str] = {}
        # Предполагается, что get_trip_and_route_alignments возвращает Dict[route_ref, List[trip_id]]
        route_alignments = self.get_trip_and_route_alignments() 
        for route_ref, trip_ids in route_alignments.items():
            for trip_id in trip_ids:
                trip_to_route_map[trip_id] = route_ref

        # Разделяем маршруты на приоритетные и остальные
        priority_routes_items: List[Tuple[str, List[int]]] = []
        other_routes_items: List[Tuple[str, List[int]]] = []

        # Создаем множество trip_ref, которые уже были добавлены из override_order
        processed_trip_ids = set()

        for route_ref_priority in override_order:
            for trip_id, route_sequence in routes_dict.items():
                if trip_to_route_map.get(trip_id) == route_ref_priority and trip_id not in processed_trip_ids:
                    priority_routes_items.append((trip_id, route_sequence))
                    processed_trip_ids.add(trip_id)
        
        # Добавляем остальные маршруты, которые не были в override_order или уже обработаны
        for trip_id, route_sequence in routes_dict.items():
            if trip_id not in processed_trip_ids:
                other_routes_items.append((trip_id, route_sequence))

        # Сортируем приоритетные маршруты по длине (длинные вперед)
        priority_routes_items.sort(key=lambda x: len(x[1]), reverse=True)
        # Сортируем остальные маршруты по длине (длинные вперед)
        other_routes_items.sort(key=lambda x: len(x[1]), reverse=True)

        # Объединяем списки маршрутов, приоритетные идут первыми
        all_routes_sorted = priority_routes_items + other_routes_items

        result_routes = {}
        unique_sequences = set() # Хранит уникальные хэшированных подпоследовательностей (как кортежи)

        for trip_id, route_sequence in all_routes_sorted:
            current_sequence = list(route_sequence) # Копия для модификации
            
            # Проверяем и удаляем подпоследовательности, которые уже были найдены
            i = 0
            while i < len(current_sequence):
                found_duplicate = False
                # Ищем самую длинную подпоследовательность, начинающуюся с i
                for j in range(len(current_sequence), i, -1):
                    subsequence = tuple(current_sequence[i:j])
                    if subsequence in unique_sequences:
                        # Если подпоследовательность уже есть, удаляем ее из текущего маршрута
                        del current_sequence[i:j]
                        found_duplicate = True
                        break
                if not found_duplicate:
                    # Если подпоследовательность уникальна, добавляем ее в множество
                    if current_sequence: # Убедиться, что список не пуст после удаления
                        unique_sequences.add(tuple(current_sequence[i:i+1]))
                    i += 1
            
            if current_sequence:
                result_routes[trip_id] = current_sequence
                # Добавляем все подпоследовательности этого маршрута (если он добавлен) в уникальные
                for i in range(len(current_sequence)):
                    for j in range(i + 1, len(current_sequence) + 1):
                        unique_sequences.add(tuple(current_sequence[i:j]))
        
        return result_routes

    def get_trip_and_route_alignments(self) -> Dict[str, List[str]]:
        """
        Формирует словарь, сопоставляющий ref маршрута с соответствующими trip_id.
        """
        refs=[]
        
        for edge in self.graph.edges():
            #print(edge)
            if edge['trip_ref'] not in refs:
                refs.append(edge['trip_ref'])
        align={ref: [] for ref in refs}
        for ref in refs:
            for edge in self.graph.edges():
                if edge['trip_ref']==ref and edge['trip_id'] not in align[ref]:
                    align[ref].append(edge['trip_id'])
        return align
    
    def get_ref_by_trip_id(self, trip_id: str) -> Optional[str]:
        """Получает ref маршрута по trip_id."""
        for ref, trip_ids in self.trips_dict.items():
            if trip_id in trip_ids:
                return ref
        return None
    
    def get_trip_id_by_ref(self, ref: str) -> List[str]:
        """Получает список trip_id по ref маршрута."""
        return self.trips_dict.get(ref, [])
            
    def get_node_idx_by_osmid(self, osmid: Any) -> Optional[int]:
        """Получает индекс узла rustworkx по OSM ID остановки."""
        for node_idx in self.graph.node_indices():
            if self.graph[node_idx].get('id') == osmid:
                return node_idx
        return None

    def get_edges_of_stoplist(self, stoplist: List[int], trip_id: str) -> List[Dict[str, Any]]:
        """
        Получает список ребер, соответствующих заданной последовательности остановок для конкретного trip_id.
        """
        edges_list = []
        for i in range(1, len(stoplist)):
            from_node_idx, to_node_idx = stoplist[i-1], stoplist[i]
            # Получаем все ребра между этими двумя узлами
            candidate_edges = self.graph.get_all_edge_data(to_node_idx, from_node_idx)
            found_edge = None
            for edge_data in candidate_edges:
                if isinstance(edge_data, dict) and edge_data.get('trip_id') == trip_id:
                    found_edge = edge_data
                    break
            if found_edge:
                edges_list.append(found_edge)
            else:
                print(f"Предупреждение: Не найдено ребро между {from_node_idx} и {to_node_idx} для trip_id {trip_id}.")
        return edges_list 
        
    def get_route_stoplist_from(self, start_node: int, trip_id: str, cutoff: int = 20) -> List[int]:
        """
        Отслеживает последовательность остановок маршрута от начального узла назад по графу
        до заданного ограничения времени в пути.

        :param start_node: Индекс узла rustworkx, с которого начинается отслеживание.
        :param trip_id: ID маршрута для отслеживания.
        :param cutoff: Максимальное совокупное время в пути (в минутах) для отслеживания.
        :return: Список индексов узлов остановок в обратном порядке (от начальной точки до точки отсечения).
        """
        node_sequence = [start_node]
        current_node = start_node
        current_traveltime = 0.0

        while True:
            found_next_node = False
            # Ищем входящие ребра, принадлежащие текущему trip_id
            in_edges_for_trip = [
                (u, v, data) for u, v, data in self.graph.in_edges(current_node) 
                if isinstance(data, dict) and data.get('trip_id') == trip_id
            ]

            if not in_edges_for_trip:
                break # Больше нет входящих ребер для этого trip_id

            # В этом случае предполагаем, что есть только одно ребро, ведущее назад по маршруту.
            # Если есть несколько, это может указывать на сложный граф или необходимость более точной логики.
            # Для простоты берем первое попавшееся.
            prev_u, _, prev_edge_data = in_edges_for_trip[0]
            
            if current_traveltime + prev_edge_data.get('traveltime', 0) > cutoff:
                break # Превышено ограничение времени в пути

            node_sequence.append(prev_u)
            current_traveltime += prev_edge_data.get('traveltime', 0)
            current_node = prev_u
            found_next_node = True

            if not found_next_node:
                break # Если не смогли найти следующую станцию

        return node_sequence
    
    def show_report_all(self, stoplists: Dict[str, List[int]]):
        """
        Генерирует и отображает объединенные карты маршрутов и станций для всех
        затронутых маршрутов.
        """
        print(stoplists.keys())
        segmentlists = {trip_id: self.get_edges_of_stoplist(nodes_list, trip_id) 
                        for trip_id, nodes_list in stoplists.items()}
        
        all_routes_gdf_list = []
        all_stations_gdf_list = []

        for trip_id, nodes_list in stoplists.items():
            current_ref = self.get_ref_by_trip_id(trip_id)
            if not current_ref:
                print(f"Предупреждение: Не удалось найти 'ref' для trip_id: {trip_id}. Пропускается.")
                continue

            print(f'### Маршрут {current_ref} (trip_id: {trip_id})')
            
            # Станции для текущего маршрута
            current_stations_data = []
            if len(nodes_list) > 1:
                for i, node_idx in enumerate(nodes_list, start=1):
                    node_data = self.graph.get_node_data(node_idx)
                    if node_data and 'name' in node_data:
                        print(f'{node_data["name"]}')
                        # Подготовка данных для GeoDataFrame
                        station_copy = node_data.copy()
                        station_copy['node_idx'] = node_idx
                        # Используем уже разобранную геометрию
                        if 'shapely_geom' in station_copy:
                             station_copy['geometry'] = station_copy.pop('shapely_geom')
                        else:
                             station_copy['geometry'] = wkt_loads(station_copy.get('geom')) # Fallback
                        current_stations_data.append(station_copy)
            
            if current_stations_data:
                stations_gdf_temp = gpd.GeoDataFrame(current_stations_data)
                stations_gdf_temp.set_geometry('geometry', crs='EPSG:4326', inplace=True)
                all_stations_gdf_list.append(stations_gdf_temp)

            # Ребра для текущего маршрута
            current_route_geom_data = []
            for edge_data in segmentlists[trip_id]:
                # Используем уже разобранную геометрию
                edge_data_copy = edge_data.copy()
                if 'geom' in edge_data_copy and isinstance(edge_data_copy['geom'], shapely.geometry.LineString):
                    edge_data_copy['geometry'] = edge_data_copy['geom'] # Уже shapely объект
                else: # Fallback, если 'geom' не shapely объект
                    edge_data_copy['geometry'] = wkt_loads(edge_data_copy.get('geom', 'LINESTRING EMPTY')) # Fallback
                current_route_geom_data.append(edge_data_copy)

            if current_route_geom_data:
                route_gdf_temp = gpd.GeoDataFrame(current_route_geom_data)
                route_gdf_temp.set_geometry('geometry', crs='EPSG:4326', inplace=True)
                all_routes_gdf_list.append(route_gdf_temp)
        
        if not all_routes_gdf_list or not all_stations_gdf_list:
            print("Нет данных для отображения отчета.")
            return

        route_gdf_combined = gpd.GeoDataFrame(pd.concat(all_routes_gdf_list, ignore_index=True))
        stations_gdf_combined = gpd.GeoDataFrame(pd.concat(all_stations_gdf_list, ignore_index=True))
        #print(route_gdf_combined.head())
        route_refs = route_gdf_combined['trip_ref'].unique()
        for route_ref in route_refs:
            # Преобразование в CRS для буферизации
            route_gdf_ref = route_gdf_combined[route_gdf_combined['trip_ref'] == route_ref].copy()
            
            if route_gdf_ref.empty:
                continue

            # Проекция для буферизации (например, UTM Zone 36N for Berlin area, or any appropriate local CRS)
            # Замените на подходящий EPSG код для вашей области
            # EPSG:32636 - UTM zone 36N (используется в оригинале)
            # EPSG:3857 - Web Mercator (хорошо для тайлов, но искажает расстояния)
            # Выберем более универсальный подход или требуем явного указания
            
            # Для общей практики, если нет специфического местного CRS,
            # иногда используют буфер в градусах для маленьких расстояний, но это не точно
            # Лучше всего использовать подходящий проекционный CRS
            
            # Попробуем 32636 (UTM zone 36N) как в оригинале
            try:
                route_gdf_ref_proj = route_gdf_ref.to_crs(epsg=32636)
                stations_gdf_combined_proj = stations_gdf_combined.to_crs(epsg=32636)

                # Буфер и фильтрация станций
                buffered_route_union = route_gdf_ref_proj.unary_union.buffer(50) # 50 метров буфер
                stations_gdf_ref = stations_gdf_combined_proj[stations_gdf_combined_proj.within(buffered_route_union)].copy()

                # Возвращаем в EPSG:4326 для отображения на тайлах
                route_gdf_ref.to_crs('EPSG:4326', inplace=True)
                stations_gdf_ref.to_crs('EPSG:4326', inplace=True)

            except Exception as e:
                print(f"Предупреждение: Не удалось перепроецировать для буферизации (возможно, неподходящий CRS или ошибка): {e}. Попытка буфера в градусах.")
                # Если перепроецирование не удалось, используем буфер в градусах (менее точный)
                route_gdf_ref_copy = route_gdf_ref.copy()
                buffered_route_union = route_gdf_ref_copy.unary_union.buffer(0.0005) # ~50m в градусах на экваторе
                stations_gdf_ref = stations_gdf_combined[stations_gdf_combined.within(buffered_route_union)].copy()


            bounds = shapely.geometry.box(*route_gdf_ref.total_bounds).buffer(0.005).bounds
            
            # Базовая карта CustomTiles
            class CustomTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = MB_URL
                    return url

            custom_tiles = CustomTiles()

            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': custom_tiles.crs})
            ax.add_image(custom_tiles, 13) # Уровень масштабирования
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())
            
            route_color = self.tram_colors.get(route_ref, self.tram_colors.get("Default", "#000000"))
            print(f"Визуализация маршрута {route_ref} с цветом {route_color}")

            # Отображение станций (белый кружок под цветным)
            stations_gdf_ref.to_crs(custom_tiles.crs).plot(ax=ax, markersize=10, color='white', zorder=3)
            stations_gdf_ref.to_crs(custom_tiles.crs).plot(ax=ax, markersize=20, color=route_color, zorder=2)

            # Отображение маршрута
            route_gdf_ref.to_crs(custom_tiles.crs).plot(ax=ax, color=route_color, linewidth=5, zorder=1)
            
            plt.title(f"Маршрут: {route_ref}")
            plt.show()

    def show_report_each(self, stoplists: Dict[str, List[int]]):
        """
        Генерирует и отображает отдельные карты маршрутов и станций для каждого
        затронутого маршрута.
        """
        segmentlists = {trip_id: self.get_edges_of_stoplist(nodes_list, trip_id) 
                        for trip_id, nodes_list in stoplists.items()}
        
        for trip_id, nodes_list in stoplists.items():
            current_ref = self.get_ref_by_trip_id(trip_id)
            if not current_ref:
                print(f"Предупреждение: Не удалось найти 'ref' для trip_id: {trip_id}. Пропускается.")
                continue

            print(f'### Маршрут {current_ref} (trip_id: {trip_id})')
            stations_for_plot = []
            if len(nodes_list) > 1:
                for i, node_idx in enumerate(nodes_list, start=1):
                    node_data = self.graph.get_node_data(node_idx)
                    if node_data and 'name' in node_data:
                        print(f'{i}. {node_data["name"]}')
                        # Подготовка данных для GeoDataFrame
                        station_copy = node_data.copy()
                        station_copy['node_idx'] = node_idx
                        if 'shapely_geom' in station_copy:
                             station_copy['geometry'] = station_copy.pop('shapely_geom')
                        else:
                             station_copy['geometry'] = wkt_loads(station_copy.get('geom')) # Fallback
                        stations_for_plot.append(station_copy)
            
            if not stations_for_plot:
                print(f"Нет станций для отображения для маршрута {current_ref}.")
                continue

            stations_gdf = gpd.GeoDataFrame(stations_for_plot)
            stations_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)

            route_geom_data = []
            for edge_data in segmentlists[trip_id]:
                edge_data_copy = edge_data.copy()
                if 'geom' in edge_data_copy and isinstance(edge_data_copy['geom'], shapely.geometry.LineString):
                    edge_data_copy['geometry'] = edge_data_copy['geom'] # Уже shapely объект
                else:
                    edge_data_copy['geometry'] = wkt_loads(edge_data_copy.get('geom', 'LINESTRING EMPTY')) # Fallback
                route_geom_data.append(edge_data_copy)

            if not route_geom_data:
                print(f"Нет ребер для отображения для маршрута {current_ref}.")
                continue

            route_gdf = gpd.GeoDataFrame(route_geom_data)
            route_gdf.set_geometry('geometry', crs='EPSG:4326', inplace=True)

            # Определение границ для карты
            bounds = shapely.geometry.box(*route_gdf.total_bounds).buffer(0.005).bounds
            
            class CustomTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = MB_URL
                    return url
            custom_tiles = CustomTiles()

            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': custom_tiles.crs})
            ax.add_image(custom_tiles, 13)
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())
            
            route_color = self.tram_colors.get(current_ref, self.tram_colors.get("Default", "#000000"))
            
            stations_gdf.to_crs(custom_tiles.crs).plot(ax=ax, markersize=10, color='white', zorder=3)
            stations_gdf.to_crs(custom_tiles.crs).plot(ax=ax, markersize=20, color=route_color, zorder=2)
            route_gdf.to_crs(custom_tiles.crs).plot(ax=ax, color=route_color, linewidth=5, zorder=1)
            
            plt.title(f"Маршрут: {current_ref}")
            plt.show()