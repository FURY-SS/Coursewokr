import open3d as o3d
import numpy as np

# создает облако точек используя файл txt
def load_coordinates_from_txt(file_path):

    # Создание пустого списка для хранения координат
    points = []

    # Открываем файл для чтения
    with open(file_path, 'r') as file:

        for line in file:

            # Убираем лишние пробелы и символы новой строки
            line = line.strip()

            # Разбиваем строку по запятым
            if line:
                x, y, z = map(float, line.split(','))
                points.append([x, y, z])

    # Преобразуем список точек в массив numpy
    points = np.array(points)

    # Создаем объект PointCloud из библиотеки open3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

# функция визуализации облака точек
def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

# Функция для сохранения облака точек в txt файл
def save_point_cloud_to_txt(point_cloud, output_file_path):
    # Получаем массив numpy из объекта PointCloud
    points = np.asarray(point_cloud.points)

    # Открываем файл для записи
    with open(output_file_path, 'w') as file:
        # Записываем каждую точку в строку в формате x, y, z
        for point in points:
            file.write(f"{point[0]},{point[1]},{point[2]}\n")

# Функция для сегментации плоскости из облака точек
def segment_plane(point_cloud, distance_threshold=0.2, ransac_n=10, num_iterations=100):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,ransac_n=ransac_n,num_iterations=num_iterations)

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    return inlier_cloud, outlier_cloud

# Функция для кластеризации облака точек
def cluster_point_cloud(point_cloud, eps=0.3, min_points=10):
    # Применяем DBSCAN для кластеризации
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # Находим количество кластеров
    max_label = labels.max()
    print(f"Количество кластеров: {max_label + 1}")

    # Инициализация для хранения данных о самом большом кластере
    largest_cluster_size = 0
    largest_cluster_cloud = None

    # Проходим по каждому кластеру
    for i in range(max_label + 1):
        # Выбираем точки, принадлежащие текущему кластеру
        cluster_indices = np.where(labels == i)[0]
        cluster_cloud = point_cloud.select_by_index(cluster_indices)

        # Оцениваем размер кластера
        cluster_size = len(cluster_indices)
        print(f"Кластер {i}: {cluster_size} точек")

        # Обновляем, если текущий кластер больше предыдущего
        if cluster_size > largest_cluster_size:
            largest_cluster_size = cluster_size
            largest_cluster_cloud = cluster_cloud

    # Проверяем наличие самого большого кластера и создаем описание
    if largest_cluster_cloud is not None:
        cluster_info = f"Самый большой кластер содержит {largest_cluster_size} точек."
        print(cluster_info)
    else:
        cluster_info = "Кластеры не найдены."
        print(cluster_info)

    return largest_cluster_cloud, cluster_info

# Выходной файл
#output_file = "out.txt"
# Входной файл
#file_path = "wreck1.txt"

# Загружаем облако точек из файла
point_cloud = load_coordinates_from_txt(file_path)

# Сегментация плоскости
inlier_cloud, outlier_cloud = segment_plane(point_cloud)

# Кластеризация
largest_cluster, cluster_info = cluster_point_cloud(outlier_cloud)

# Сохранени облака в файл
save_point_cloud_to_txt(largest_cluster, output_file)

# Вывод информации о кластере
#print(cluster_info)

#Визуализация самого большого кластера
#if largest_cluster is not None:
    #visualize_point_cloud(largest_cluster)
#else:
    #print("Нет кластера для визуализации.")
