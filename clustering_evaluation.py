import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import geohash
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import os
import time


class FoursquareDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_raw_data(self):
        """加载原始Foursquare数据集"""
        columns = ['user_id', 'poi_id', 'poi_category_id', 'poi_category_name',
                   'latitude', 'longitude', 'timezone_offset', 'timestamp']

        try:
            # 尝试不同的编码格式
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    df = pd.read_csv(self.data_path, sep='\t', header=None,
                                     names=columns, encoding=encoding,
                                     on_bad_lines='skip')
                    print(f"成功加载数据集: {os.path.basename(self.data_path)} (编码: {encoding})")
                    print(f"原始数据量: {len(df)} 条签到记录")
                    return df
                except UnicodeDecodeError:
                    continue

            # 所有编码都失败，进行错误处理
            df = pd.read_csv(self.data_path, sep='\t', header=None,
                             names=columns, encoding='latin-1',
                             on_bad_lines='skip',
                             engine='python')
            print(f"使用备选方案加载数据集: {os.path.basename(self.data_path)}")
            print(f"原始数据量: {len(df)} 条签到记录")
            return df

        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def preprocess_data(self, df, min_user_checkins=10, min_poi_checkins=10):
        """数据预处理"""
        print("\n开始数据预处理...")

        # 1. 删除低频用户
        user_checkin_counts = df['user_id'].value_counts()
        valid_users = user_checkin_counts[user_checkin_counts >= min_user_checkins].index
        df = df[df['user_id'].isin(valid_users)]
        print(f"过滤低频用户后: {len(df)} 条记录")

        # 2. 删除低频POI
        poi_checkin_counts = df['poi_id'].value_counts()
        valid_pois = poi_checkin_counts[poi_checkin_counts >= min_poi_checkins].index
        df = df[df['poi_id'].isin(valid_pois)]
        print(f"过滤低频POI后: {len(df)} 条记录")

        # 3. 转换时间戳
        sample_timestamps = df['timestamp'].head(3).tolist()
        print(f"时间戳样例: {sample_timestamps}")

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S %z %Y')
        except Exception as e:
            print(f"标准格式失败: {e}")
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            invalid_count = df['timestamp'].isna().sum()
            if invalid_count > 0:
                # 将时间戳转换失败的记录删除"
                df = df.dropna(subset=['timestamp'])

        df = df.sort_values(['user_id', 'timestamp'])

        # 4. 去除重复POI信息（用于聚类）
        pois_df = df[['poi_id', 'latitude', 'longitude', 'poi_category_id']].drop_duplicates('poi_id')
        print(f"去重后: {len(pois_df)}")

        return df, pois_df

    def get_pois_for_clustering(self, pois_df):
        """获取用于聚类的POI数据"""
        clustering_data = pois_df[['poi_id', 'latitude', 'longitude']].copy()
        return clustering_data


class ClusteringEvaluator:
    def __init__(self, pois_data):
        """
        pois_data: DataFrame with columns ['poi_id', 'latitude', 'longitude']
        """
        self.pois_data = pois_data
        self.coords = pois_data[['latitude', 'longitude']].values

    def _time_method(self, method_func, *args, **kwargs):
        start_time = time.time()
        result = method_func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    def original_dbscan(self, eps_km, min_samples):
        print("=" * 20+"原始DBSCAN聚类"+"=" * 20)
        start_time = time.time()
        coords_rad = np.radians(self.coords)   # 将经纬度转换为弧度
        eps_rad = eps_km / 6371.0  # 将km转换为弧度
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
        labels = db.fit_predict(coords_rad)
        end_time = time.time()
        elapsed_time = end_time - start_time

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        n_total_points = len(labels)
        print(f"📊 聚类结果统计:")
        print(f"  簇数量: {n_clusters}个")
        print(f"  噪声点: {n_noise}个 ({n_noise / n_total_points * 100:.1f}%)")
        print(f"  总点数: {n_total_points}个")
        print(f"  平均簇大小: {n_total_points / n_clusters:.1f}点/簇" if n_clusters > 0 else "  平均簇大小: N/A")
        print(f"  执行时间: {elapsed_time:.3f}秒")
        return labels, elapsed_time

    def geohash_dbscan(self, eps_km, min_samples, geohash_precision):
        """GeoHash + DBSCAN方法 """
        print("=" * 20 + "GeoHash + DBSCAN方法" + "=" * 20)
        start_time = time.time()
        pois_data_reset = self.pois_data.reset_index(drop=True)  # 重置索引确保连续性
        coords_rad_global = np.radians(self.coords)  # 将全局坐标转换为弧度
        geohash_to_pois = defaultdict(list)  # GeoHash编码分组

        for idx, row in pois_data_reset.iterrows():
            gh = geohash.encode(row['latitude'], row['longitude'], geohash_precision)
            geohash_to_pois[gh].append((idx, row['latitude'], row['longitude']))

        print(f"GeoHash划分区域数量: {len(geohash_to_pois)}")
        all_labels = np.full(len(pois_data_reset), -1)  # 初始化标签数组
        cluster_id_offset = 0

        eps_rad = eps_km / 6371.0  # 将km转换为弧度
        region_stats = []
        processed_points = 0
        total_clusters_created = 0

        # 区域处理
        for i, (gh, pois_in_gh) in enumerate(geohash_to_pois.items()):
            n_points = len(pois_in_gh)

            if n_points < min_samples:
                region_stats.append((gh, n_points, 0, n_points))
                continue

            indices = [item[0] for item in pois_in_gh]  # 提取该区域的坐标（弧度）
            coords_gh_rad = coords_rad_global[indices]  # 使用全局的弧度坐标

            # 在该区域内运行DBSCAN（Haversine距离）
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
            labels_gh = db.fit_predict(coords_gh_rad)

            # 统计该区域聚类情况
            unique_labels = set(labels_gh)
            n_clusters = len([l for l in unique_labels if l != -1])
            n_noise = np.sum(labels_gh == -1)
            region_stats.append((gh, n_points, n_clusters, n_noise))
            total_clusters_created += n_clusters
            processed_points += n_points

            # 重新编号并赋值标签
            for j, label in enumerate(labels_gh):
                idx = indices[j]
                if label != -1:
                    all_labels[idx] = label + cluster_id_offset

            # 更新偏移量
            if n_clusters > 0:
                max_label = max([l for l in labels_gh if l != -1])
                cluster_id_offset += (max_label + 1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        n_total_points = len(pois_data_reset)
        n_total_noise = np.sum(all_labels == -1)


        print(f"\n--总体聚类统计:")
        print(f"  簇数量: {total_clusters_created}个")
        print(f"  噪声点: {n_total_noise}个 ({n_total_noise / n_total_points * 100:.1f}%)")
        print(f"  总点数: {n_total_points}个")
        print(
            f"  平均簇大小: {processed_points / total_clusters_created:.1f}点/簇" if total_clusters_created > 0 else "  平均簇大小: N/A")
        print(f"  执行时间: {elapsed_time:.3f}秒")

        # 分析区域大小分布
        region_sizes = [s[1] for s in region_stats]
        print(f"\n--区域大小分布:")
        print(f"  最大区域: {max(region_sizes)}点")
        print(f"  最小区域: {min(region_sizes)}点")
        print(f"  平均区域: {np.mean(region_sizes):.1f}点")
        print(f"  中位数区域: {np.median(region_sizes):.1f}点")

        valid_regions = [s for s in region_stats if s[1] >= min_samples]
        print(f"有效区域数量: {len(valid_regions)}/{len(region_stats)}")

        return all_labels,elapsed_time

    def evaluate_clustering(self, labels, method_name):
        """评估聚类质量"""
        # 过滤掉噪声点（label = -1）
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_coords = self.coords[valid_mask]

        if len(set(valid_labels)) < 2:
            print(f"{method_name}: 有效聚类数少于2，无法计算指标")
            return {
                'method': method_name,
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': len(set(valid_labels)),
                'n_noise': np.sum(labels == -1),
                'coverage': len(valid_labels) / len(labels),
                'avg_cluster_size': 0
            }

        # 计算平均簇大小
        cluster_sizes = [np.sum(valid_labels == label) for label in set(valid_labels)]
        avg_cluster_size = np.mean(cluster_sizes)

        metrics = {
            'method': method_name,
            'silhouette_score': silhouette_score(valid_coords, valid_labels),
            'calinski_harabasz_score': calinski_harabasz_score(valid_coords, valid_labels),
            'davies_bouldin_score': davies_bouldin_score(valid_coords, valid_labels),
            'n_clusters': len(set(valid_labels)),
            'n_noise': np.sum(labels == -1),
            'coverage': len(valid_labels) / len(labels),
            'avg_cluster_size': avg_cluster_size
        }
        return metrics

    def compare_methods(self, eps_km=1.0, min_samples=5, geohash_precision=6):
        """比较两种聚类方法 - 使用km单位"""
        print("\n聚类质量评估...")

        # 方法1: 原始DBSCAN with Haversine
        labels_original, time_original = self.original_dbscan(eps_km, min_samples)
        metrics_original = self.evaluate_clustering(labels_original, "Original DBSCAN")
        metrics_original['execution_time'] = time_original

        # 方法2: GeoHash + DBSCAN with Haversine
        labels_geohash, time_geohash = self.geohash_dbscan(eps_km, min_samples, geohash_precision)
        metrics_geohash = self.evaluate_clustering(labels_geohash, "GeoHash+DBSCAN")
        metrics_geohash['execution_time'] = time_geohash

        # 添加results_df的定义
        results_df = pd.DataFrame([metrics_original, metrics_geohash])

        # 简单的性能对比输出
        print("=" * 20 + "性能对比总结" + "=" * 20)
        print(f"原始DBSCAN执行时间: {time_original:.3f}秒")
        print(f"GeoHash+DBSCAN执行时间: {time_geohash:.3f}秒")
        print(f"速度提升: {time_original / time_geohash:.2f}倍")

        return results_df, labels_original, labels_geohash

    def visualize_comparison(self, labels_original, labels_geohash, save_path=None):
        """可视化对比两种方法"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 原始DBSCAN结果
            scatter1 = axes[0].scatter(self.coords[:, 1],self.coords[:, 0],
                                       c=labels_original, cmap='tab10', s=8, alpha=0.7)
            axes[0].set_title('Original DBSCAN Clustering', fontsize=14)
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            plt.colorbar(scatter1, ax=axes[0])

            # GeoHash+DBSCAN结果
            scatter2 = axes[1].scatter(self.coords[:, 1],self.coords[:, 0],
                                       c=labels_geohash, cmap='tab10', s=8, alpha=0.7)
            axes[1].set_title('GeoHash + DBSCAN Clustering', fontsize=14)
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            plt.colorbar(scatter2, ax=axes[1])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果已保存至: {save_path}")

            plt.show()

        except Exception as e:
            print(f"可视化失败: {e}")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果已保存至: {save_path} (但无法显示)")
            plt.close()


def main():
    data_path = r"E:\code\lwCode\My_Code\dataset\dataset_TSMC2014_NYC.txt"  # 数据集路径

    # 1. 加载数据
    loader = FoursquareDataLoader(data_path)
    raw_df = loader.load_raw_data()

    if raw_df is None:
        print("数据加载失败，请检查文件路径和编码")
        return

    # 2. 预处理数据
    processed_df, pois_df = loader.preprocess_data(raw_df)
    clustering_data = loader.get_pois_for_clustering(pois_df)

    print(f"\n用于聚类的POI数据:")
    print(f"- POI数量: {len(clustering_data)}")
    print(f"- 经纬度范围: {clustering_data['latitude'].min():.3f}~{clustering_data['latitude'].max():.3f}, "
          f"{clustering_data['longitude'].min():.3f}~{clustering_data['longitude'].max():.3f}")

    # 3. 聚类评估 - 使用优化参数
    evaluator = ClusteringEvaluator(clustering_data)

    # 参数（使用km单位）
    eps_km = 1.0  # 1.0公里（原来0.01°≈1.1km）
    min_samples = 3
    geohash_precision = 5

    print(f"\n使用参数: eps={eps_km}, min_samples={min_samples}, geohash_precision={geohash_precision}")

    results, labels_orig, labels_geo = evaluator.compare_methods(
        eps_km=eps_km,  # 改为eps_km
        min_samples=min_samples,
        geohash_precision=geohash_precision
    )

    # 4. 显示结果
    print("\n" + "=" * 20+"聚类质量评估结果"+"=" * 20)
    display_columns = ['method', 'silhouette_score', 'calinski_harabasz_score',
                       'davies_bouldin_score', 'n_clusters', 'n_noise', 'coverage', 'avg_cluster_size']
    print(results[display_columns].round(4))

    # 5. 结果分析
    print("\n" + "=" * 20 + "改进效果" + "=" * 20)

    if not np.isnan(results.iloc[0]['silhouette_score']):
        sil_improve = (results.iloc[1]['silhouette_score'] - results.iloc[0]['silhouette_score']) / results.iloc[0][
            'silhouette_score'] * 100
        ch_improve = (results.iloc[1]['calinski_harabasz_score'] - results.iloc[0]['calinski_harabasz_score']) / \
                     results.iloc[0]['calinski_harabasz_score'] * 100
        db_improve = (results.iloc[0]['davies_bouldin_score'] - results.iloc[1]['davies_bouldin_score']) / \
                     results.iloc[0]['davies_bouldin_score'] * 100

        print(f"Silhouette Score 变化: {sil_improve:+.2f}%")
        print(f"Calinski-Harabasz Score 变化: {ch_improve:+.2f}%")
        print(f"Davies-Bouldin Score 变化: {db_improve:+.2f}% (越低越好)")
        print(f"聚类数量变化: {results.iloc[0]['n_clusters']} → {results.iloc[1]['n_clusters']}")
        print(
            f"噪声点比例变化: {results.iloc[0]['n_noise'] / len(clustering_data):.3f} → {results.iloc[1]['n_noise'] / len(clustering_data):.3f}")
        print(f"平均簇大小变化: {results.iloc[0]['avg_cluster_size']:.1f} → {results.iloc[1]['avg_cluster_size']:.1f}")

    # 6. 可视化
    save_path = f"./clustering_eps={eps_km}_minS={min_samples}_NYC_P{geohash_precision}.png"
    evaluator.visualize_comparison(labels_orig, labels_geo, save_path=save_path)


if __name__ == "__main__":
    main()