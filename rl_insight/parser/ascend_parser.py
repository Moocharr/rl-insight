import json
from loguru import logger
import os
from collections import defaultdict
from pathlib import Path

from .parser import BaseClusterParser
from rl_insight.utils.schema import Constant, DataMap


class AscendClusterParser(BaseClusterParser):
    def allocate_prof_data(self, input_path: str) -> list[DataMap]:
        ascend_pt_dirs = []
        for root, dirs, _ in os.walk(input_path):
            for dir_name in dirs:
                if dir_name.endswith(Constant.ASCEND_PROFILER_SUFFIX):
                    path = os.path.join(root, dir_name)
                    ascend_pt_dirs.append(
                        {"role": Path(path).parent.name, "path": path}
                    )
        data_map = self._get_data_map(ascend_pt_dirs)
        data_maps = self._get_rank_path_with_role(data_map)
        return data_maps

    def _get_profiler_data_path(self, rank_id, data_path):
        return os.path.join(data_path, Constant.ASCEND_PROFILER_OUTPUT)

    def _get_rank_path_with_role(self, data_map) -> list[DataMap]:
        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        rank_ids_with_role = list(data_map.keys())
        data_paths: list[DataMap] = []
        for task_role, rank_id in rank_ids_with_role:
            rank_path_list = data_map[(task_role, rank_id)]
            profiler_data_path_list = [
                self._get_profiler_data_path(rank_id, rank_path)
                for rank_path in rank_path_list
            ]
            for profiler_data_path in profiler_data_path_list:
                data_path_dict: DataMap = {
                    Constant.RANK_ID: rank_id,
                    Constant.ROLE: task_role,
                    Constant.PROFILER_DATA_PATH: "",
                }

                if os.path.exists(profiler_data_path):
                    data_path_dict[Constant.PROFILER_DATA_PATH] = profiler_data_path
                    data_paths.append(data_path_dict)
                else:
                    logger.warning(
                        f"Profiler data not found, rank id: {rank_id}, data path: {profiler_data_path}."
                    )
        return data_paths

    def _get_data_map(self, path_list) -> dict[tuple[str, int], list[str]]:
        data_map: dict[tuple[str, int], list[str]] = {}
        rank_id_map = defaultdict(list)
        for path_info in path_list:
            role = path_info.get("role")
            dir_name = path_info.get("path")
            rank_id = self._get_rank_id(dir_name)
            task_role = self._get_task_role(dir_name)
            if task_role is None:
                task_role = role
            if rank_id < 0:
                logger.error(f"direct:{dir_name} fail to get rankid or rankid invalid.")
                continue
            rank_id_map[(task_role, rank_id)].append(dir_name)
        try:
            for map_key, dir_list in rank_id_map.items():
                dir_list.sort(key=lambda x: x.split("_")[-3])
                data_map[map_key] = dir_list
        except Exception as e:
            raise RuntimeError("Found invalid directory name!") from e
        return data_map

    def _get_rank_id(self, dir_name: str) -> int:
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name.startswith(
                Constant.ASCEND_PROFILER_INFO_HEAD
            ) and file_name.endswith(Constant.JSON_EXTENSION):
                base_name = os.path.splitext(file_name)[0]
                rank_id_str = base_name[len(Constant.ASCEND_PROFILER_INFO_HEAD):]
                try:
                    rank_id = int(rank_id_str)
                except ValueError:
                    rank_id = -1
                return rank_id
        return -1

    def _get_task_role(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name == Constant.ASCEND_PROFILER_METADATA_JSON:
                with open(os.path.join(dir_name, file_name), encoding="utf-8") as f:
                    config = json.load(f)
                task_role = config.get("role")
                if task_role:
                    return task_role
        return None