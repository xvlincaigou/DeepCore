from .earlytrain import EarlyTrain
import numpy as np
from scipy import sparse
import torch
from ..nets.nets_utils import MyDataParallel
import networkx as nx
import faiss
from .methods_utils.special_select import special_select
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from .methods_utils.euclidean import euclidean_dist
from .kcentergreedy import k_center_greedy


class Graphcentrality(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, 
                 specific_model=None, torchvision_pretrain: bool = True, 
                 dst_pretrain_dict: dict = {}, fraction_pretrain=1, dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, 
                         torchvision_pretrain, dst_pretrain_dict, fraction_pretrain, dst_test, **kwargs)

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features
        
    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                matrix = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                    torch.utils.data.Subset(self.dst_train, index),
                                    batch_size=self.args.selection_batch,
                                    num_workers=self.args.workers)

                for _, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0)
    
    def build_naive_graph(self):
        hidden_representations = self.construct_matrix().cpu().numpy()
        n_neighbours = 10
        
        print('Using faiss for nearest neighbours search...')
        # 使用FAISS
        index = faiss.IndexFlatL2(hidden_representations.shape[1])
        index.add(hidden_representations.astype('float32'))
        print('Searching for nearest neighbours...')
        
        # 批量搜索
        batch_size = 1000
        n_samples = len(hidden_representations)
        rows_list = []
        cols_list = []
        dist_list = []
        
        print('Computing the nearest neighbours...')
        for i in range(0, n_samples, batch_size):
            print('Processing batch', i // batch_size + 1, 'out of', n_samples // batch_size)
            end_idx = min(i + batch_size, n_samples)
            batch = hidden_representations[i:end_idx].astype('float32')
            
            D, I = index.search(batch, n_neighbours)
            # batch_rows的形状是[n_neighbours*(end_idx-i),]，比如[0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
            batch_rows = np.repeat(np.arange(i, end_idx), n_neighbours)
            rows_list.append(batch_rows)
            # 把D展平，原来是[[0,0,0],[1,1,1],[2,2,2],...]，现在是[0,0,0,1,1,1,2,2,2,...]
            cols_list.append(I.reshape(-1))
            dist_list.append(D.reshape(-1))
            
            del I, D
            import gc
            gc.collect()
            
        print('Storing the results...')
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        dists = np.concatenate(dist_list)
        connection = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)), shape=(n_samples, n_samples)
        )
        distance = sparse.csr_matrix(
            (dists, (rows, cols)), shape=(n_samples, n_samples)
        )
        
        connection = connection + connection.T
        diag_values = connection.diagonal() / 2
        connection.setdiag(0)
        connection = connection + sparse.diags(diag_values)
        
        distance = distance + distance.T
        diag_values = distance.diagonal() / 2
        distance.setdiag(0)
        distance = distance + sparse.diags(diag_values)
        
        print('Plotting the graph...')
        
        position_2d = PCA(n_components=2).fit_transform(hidden_representations)
        node_positions = {i: position_2d[i].tolist() for i in range(n_samples)}
                
        return connection, distance, n_samples, node_positions
    
    def plot_graph(self, G, selected_indices, centrality_name, fraction, node_positions):

        plt.figure(figsize=(24, 16))
        pos = node_positions

        # 创建一个数组，表示每个节点的颜色（1 表示选中，0 表示未选中）
        n_samples = G.number_of_nodes()
        node_colors = np.zeros(n_samples)
        node_colors[selected_indices] = 1  # 被选中的节点设为 1

        # 定义颜色映射：0 对应浅色，1 对应深色
        cmap = ListedColormap(['lightgray', 'darkblue'])

        # 绘制节点，颜色根据是否被选中
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=5,
            cmap=cmap,
            node_color=node_colors,
            alpha=0.7
        )

        cbar = plt.colorbar(nodes, label=f'Selected (fraction={fraction})', ticks=[0, 1])
        cbar.ax.set_yticklabels(['Not Selected', 'Selected'])  # 设置刻度标签

        plt.axis('off')
        plt.savefig(centrality_name + '-' + str(fraction) + '.png')
        plt.close()

class Degreecentrality(Graphcentrality):
    def select(self, **kwargs):
        
        self.run()
        
        connection, distance, _, node_positions = self.build_naive_graph()
        
        print('Computing the degree centrality...')
        # connection.sum(axis=1)是一个(n_samples, 1)的数组，表示每个样本的度数，现在把它展平
        degrees = np.array(connection.sum(axis=1)).flatten()
        # 默认是升序排列，所以用-num_selected取最后的部分
        idx = np.argsort(degrees)
        idx = idx[::-1]
        num_selected = int(self.fraction * len(degrees))
        indices = special_select(idx, num_selected, connection, distance)
        coreset = {"indices": indices}
        
        self.plot_graph(nx.from_scipy_sparse_array(distance), indices, 'degree_centrality', self.fraction, node_positions)
        
        return coreset

class Eigenvectorcentrality(Graphcentrality):
    def select(self, **kwargs):

        self.run()

        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Computing the eigenvector centrality...')
        G = nx.from_scipy_sparse_array(connection)
        try:
            eigenvector_dict = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=50000)
            eigenvector_scores = np.array([eigenvector_dict[i] for i in range(n_samples)])
            idx = np.argsort(eigenvector_scores)
            idx = idx[::-1]
            num_selected = int(self.fraction * len(eigenvector_scores))
            indices = special_select(idx, num_selected, connection, distance)
            coreset = {"indices": indices}
            
            self.plot_graph(G, indices, 'eigenvector_centrality', self.fraction, node_positions)
            
            return coreset
        except nx.AmbiguousSolution:
            print("Error: The graph G is not connected, so the solution is ambiguous.")
            return {"indices": []}
        except nx.ArpackNoConvergence as e:
            print("Error: Convergence not achieved. Eigenvalues and eigenvectors so far:")
            print("Eigenvalues:", e.eigenvalues)
            print("Eigenvectors:", e.eigenvectors)
            return {"indices": []}
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            return {"indices": []}

class Katzcentrality(Graphcentrality):
    def select(self, **kwargs):
        
        self.run()
        
        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Computing the Katz centrality...')
        G = nx.from_scipy_sparse_array(connection)
        try:
            katz_dict = nx.katz_centrality_numpy(G, weight='weight')
            katz_scores = np.array([katz_dict[i] for i in range(n_samples)])
            idx = np.argsort(katz_scores)
            idx = idx[::-1]
            num_selected = int(self.fraction * len(katz_scores))
            indices = special_select(idx, num_selected, connection, distance)
            coreset = {"indices": indices}
            
            self.plot_graph(G, indices, 'katz_centrality', self.fraction, node_positions)
            
            return coreset
        except nx.PowerIterationFailedConvergence:
            print("Error: Power iteration failed to converge.")
            return {"indices": []}
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            return {"indices": []}

class Closenesscentrality(Graphcentrality):
    def select(self, **kwargs):
        
        self.run()
        
        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Computing the closeness centrality...')
        G = nx.from_scipy_sparse_array(distance)
        closeness_dict = nx.closeness_centrality(G, distance='weight')
        closeness_scores = np.array([closeness_dict[i] for i in range(n_samples)])
        idx = np.argsort(closeness_scores)
        idx = idx[::-1]
        num_selected = int(self.fraction * len(closeness_scores))
        indices = special_select(idx, num_selected, connection, distance)
        coreset = {"indices": indices}
        
        self.plot_graph(G, indices, 'closeness_centrality', self.fraction, node_positions)
        
        return coreset

class Betweennesscentrality(Graphcentrality):
    def select(self, **kwargs):
        
        self.run()
        
        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Computing the betweenness centrality...')
        G = nx.from_scipy_sparse_array(distance)
        betweenness_dict = nx.betweenness_centrality(G, weight='weight')
        betweenness_scores = np.array([betweenness_dict[i] for i in range(n_samples)])
        idx = np.argsort(betweenness_scores)
        idx = idx[::-1]
        num_selected = int(self.fraction * len(betweenness_scores))
        indices = special_select(idx, num_selected, connection, distance)
        coreset = {"indices": indices}
        
        self.plot_graph(G, indices, 'betweenness_centrality', self.fraction, node_positions)
        
        return coreset
       
class BlueNoise(Graphcentrality):
    def select(self, **kwargs):
        
        self.run()
        
        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Computing the blue noise sampling...')
        idx = np.array(list(range(n_samples)))
        num_selected = int(self.fraction * len(idx))
        indices = special_select(idx, num_selected, connection, distance)
        
        coreset = {"indices": indices}
        
        self.plot_graph(nx.from_scipy_sparse_array(distance), indices, 'blue_noise', self.fraction, node_positions)
        
        return coreset
    
class DegreeKCenter(Graphcentrality):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=0,
                 specific_model="ResNet18", balance: bool = False, already_selected=[], 
                 metric="euclidean", torchvision_pretrain: bool = True, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, **kwargs)
        
        self.balance = balance
        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)
        
        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            
        self.specific_model = specific_model
        self.torchvision_pretrain = torchvision_pretrain
        self.epochs = epochs
        
    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                matrix = []
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.args.selection_batch,
                    num_workers=self.args.workers
                )
                
                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)
                    
        self.model.no_grad = False
        return torch.cat(matrix, dim=0)

    def select(self, **kwargs):
        self.run()
        
        # 第一步：使用degree centrality选择sqrt(fraction)的样本
        connection, distance, n_samples, node_positions = self.build_naive_graph()
        
        print('Step 1: Computing the degree centrality...')
        degrees = np.array(connection.sum(axis=1)).flatten()
        idx = np.argsort(degrees)[::-1]  # 降序排列
        
        # 计算第一阶段要选择的样本数量（sqrt(fraction)）
        first_stage_fraction = np.sqrt(self.fraction)
        first_stage_num = int(first_stage_fraction * n_samples)
        first_stage_indices = idx[:first_stage_num]
        
        self.plot_graph(
            nx.from_scipy_sparse_array(distance), 
            first_stage_indices,
            'degree_kcenter_1', 
            self.fraction, 
            node_positions
        )
        
        print(f'Selected {first_stage_num} samples in first stage')
        
        # 第二步：在第一步选择的样本中使用k-center-greedy选择最终样本
        print('Step 2: Applying k-center-greedy on selected samples...')
        
        second_stage_fraction = self.fraction / first_stage_fraction
        
        if self.balance:
            final_selection = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                # 在第一阶段选择的样本中筛选出属于当前类别的样本
                first_stage_indices = first_stage_indices.copy()
                class_mask = self.dst_train.targets[first_stage_indices] == c
                class_indices = first_stage_indices[class_mask]
                
                if len(class_indices) > 0:
                    final_selection = np.append(
                        final_selection,
                        k_center_greedy(
                            self.construct_matrix(class_indices),
                            budget=round(second_stage_fraction * len(class_indices)),
                            metric=self.metric,
                            device=self.args.device,
                            random_seed=self.random_seed,
                            index=class_indices,
                            already_selected=self.already_selected[np.in1d(self.already_selected, class_indices)],
                            print_freq=self.args.print_freq
                        )
                    )
        else:
            # 构建第一阶段选择样本的特征矩阵
            first_stage_matrix = self.construct_matrix(first_stage_indices)
            final_num = int(self.fraction * n_samples)
            
            # 在第一阶段选择的样本中应用k-center-greedy
            final_subset_indices = k_center_greedy(
                first_stage_matrix,
                budget=final_num,
                metric=self.metric,
                device=self.args.device,
                random_seed=self.random_seed,
                already_selected=self.already_selected[np.in1d(self.already_selected, first_stage_indices)],
                print_freq=self.args.print_freq
            )
            
            # 将子集索引映射回原始索引
            final_selection = first_stage_indices[final_subset_indices]
        
        print(f'Selected {len(final_selection)} samples in total')
        
        # 绘制最终选择结果的图
        self.plot_graph(
            nx.from_scipy_sparse_array(distance), 
            final_selection, 
            'degree_kcenter_2', 
            self.fraction, 
            node_positions
        )
        
        return {"indices": final_selection}