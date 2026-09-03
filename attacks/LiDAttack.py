import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import time

class LiDAttack():
    """
    Black-box Attack on point cloud using LiDAttack.
    Paper: https://arxiv.org/pdf/2411.01889
    This code is based on their implementation: https://github.com/Cinderyl/LiDAttack
    The main difference to their implementation is that we converted most steps to pytorch instead of using numpy.
    """
    def __init__(self, fitness_threshold=0.5, max_iterations=100, population_size=10, mutation_rate=0.1, verbose=0):
        self.fitness_threshold = fitness_threshold
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.verbose = verbose

    def get_fitness(self, perturbation_points):
        # add perturbation points to base point cloud
        new_points = torch.cat((self.base_points, perturbation_points), axis=0)

        # prepare for testing
        self.data['points'][0][0] = new_points.float()

        # get results
        results = self.model.predict(return_loss=False, rescale=True, **self.data)
        scores = results[0]["pts_bbox"]["scores_3d"]
        confidence = scores[scores >= 0.1]

        # return the average confidence
        return torch.mean(confidence)

    # selection function
    def selection(self, fitnesses, num_parents, device):
        # choose first num_parents based on highest fitness
        parents = torch.zeros(
            (num_parents, self.population.shape[1]),
            device=self.population.device,
            dtype=self.population.dtype
        )

        # find parents with highest fitness
        parent_fitness, indices = torch.topk(fitnesses, num_parents, largest=True)
        parents = self.population[indices]

        return parents, parent_fitness

    # crossover function
    def crossover(self, parents, parent_fitness, device):
        num_offspring = self.population_size - parents.shape[0]
        D = self.population.shape[1] # in their code they use the perturbation shape, but I believe it should be the population shape. (Otherwise would return a dim of 5, the number of fields each point has)

        offspring = torch.zeros(
            (num_offspring, D, self.perturbation.shape[1]),
            device=parents.device,
            dtype=parents.dtype
        )
        num_parents = parents.shape[0]

        for i in range(num_offspring):
            # randomly choose the parents for recombination
            parent1_index = torch.randint(0, num_parents, (1,), device=parents.device).item()
            parent2_index = torch.randint(0, num_parents, (1,), device=parents.device).item()

            # ensure parent1 != parent2
            while parent2_index == parent1_index:
                parent2_index = torch.randint(0, num_parents, (1,), device=parents.device).item()

            # crossover points
            crossover_point = torch.randint(1, D - 1, (1,), device=parents.device).item()

            # select parents
            p1 = parents[parent1_index]
            p2 = parents[parent2_index]

            # This part caused issues so logic had to be changed slightly! Also point cloud format related changes
            # Now I choose based on fitness
            if parent_fitness[parent1_index] > parent_fitness[parent2_index]:
                offspring[i, :crossover_point, :] = p1[:crossover_point, :]
                offspring[i, crossover_point:, :] = p2[crossover_point:, :]
            else:
                offspring[i, :crossover_point, :] = p2[:crossover_point, :]
                offspring[i, crossover_point:, :] = p1[crossover_point:, :]

        return offspring

    # mutation function
    def mutation(self, offspring_crossover, device):
        """
        Differs significantly from the version provided in the repository, they compute a number of mutations across the entire population,
        then apply it to every individal fully!
        """
        # go through all offsprings
        for i in range(0, offspring_crossover.shape[0]):
            # Bernoulli mask: True where mutation occurs
            mutation_mask = torch.rand((self.perturbation.shape[0],self.perturbation.shape[1]), device=device) < self.mutation_rate

            # Apply mutation only to selected genes
            offspring_crossover[i, mutation_mask] += (
                torch.randn(mutation_mask.sum(), device=device) * 0.05
            )

        return offspring_crossover

    def evolve(self, perturbation, device):
        total_start = time.time()
        start = time.time()
        self.perturbation = perturbation
        # vertical stack of pointclouds
        self.population = perturbation.unsqueeze(0).repeat(self.population_size, 1, 1)

        fitnesses = torch.zeros(self.population_size, device=device, dtype=self.population.dtype)

        # repeat loop until attack is good enough or the max iterations is reached
        for i in range(self.max_iterations):
            # Compute fitness for each individual
            for j in range(self.population_size):
                fitness_val = self.get_fitness(self.population[j, :])
                fitnesses[j] = fitness_val
                self.population[j, -1] = fitness_val  # store in last column

            # Print best fitness
            best_fitness_val = fitnesses.max().item()
            if self.verbose > 2 and (i%50==0 or i==self.max_iterations-1):
                print(f'Iteration: {i + 1}, Best fitness: {best_fitness_val:.6f}')
            # end = time.time()
            # print("fitness: ", end - start, "s")
            # start = time.time()

            # break condition: if fitness value is good enough or the max iterations are reached
            if best_fitness_val >= self.fitness_threshold or i == self.max_iterations - 1:
                break

            # genetic manipulation
            num_parents = 2
            parents, parent_fitness = self.selection(fitnesses, num_parents, device)
            # end = time.time()
            # print("selection: ", end - start, "s")
            # start = time.time()
            offspring_crossover = self.crossover(parents, parent_fitness, device)
            # end = time.time()
            # print("crossover: ", end - start, "s")
            # start = time.time()
            offspring_mutation = self.mutation(offspring_crossover, device)
            # end = time.time()
            # print("mutation: ", end - start, "s")
            # start = time.time()
            # Update population
            new_population = torch.empty_like(self.population)
            # elitism: keep best parents
            new_population[:num_parents] = parents
            # fill the rest with offspring
            new_population[num_parents:] = offspring_mutation
            self.population = new_population
            # end = time.time()
            # print("end loop: ", end - start, "s")
            # start = time.time()

        # return the optimal perturbation
        index = torch.argmax(fitnesses).item()
        best_perturbation = self.population[index, :]
        if self.verbose >= 3:
            print('Best perturbation:', best_perturbation)

        # return the optimal fitness
        best_fitness = self.get_fitness(best_perturbation)
        if self.verbose > 2:
            print('Best fitness:', best_fitness)
        # end = time.time()
        # print("------ Evolution: ", end - total_start, "s ------")
        return best_perturbation

    def attack(self, data, model, gt_bboxes3d, gt_labels_3d, device):
        """
        Preparing and executing iterative evolution algorithm.
        """
        self.model = model
        self.data = data
        # Extract points
        points = data['points'][0][0].to(device).clone().detach()
        adv_pc = copy.deepcopy(points) # TODO: might be unnecessary to clone again, but I am too lazy to check
        self.base_points = points
        # Genereate random initial perturbations and execute evolution
        # start = time.time()
        objects = self.extraction(points, device)
        # end = time.time()
        # Instead of extracting one object and evolving per object we evolve per scene to save ressources
        perturbation_list = []
        for obj in objects:
            if obj is None:
                continue
            init_perturbation = self.add_noise(obj)
            perturbation_list.append(init_perturbation)

        perturbation_points = torch.cat(perturbation_list, dim=0)
        perturbation = self.evolve(perturbation_points, device)
        adv_pc = torch.cat((adv_pc, perturbation), axis=0)

        data['points'][0][0] = adv_pc.float()
        result = model.predict(return_loss=False, rescale=True, **data)
        return result, adv_pc

    def add_noise(self, point_cloud, std=0.1):
        """
        Adding noise to a point cloud.
        Taken from: https://github.com/Cinderyl/LiDAttack/blob/main/GA/5.%E5%9C%A8%E7%82%B9%E4%BA%91%E8%BE%B9%E7%95%8C%E4%B8%8A%E4%BA%A7%E7%94%9F%E9%9A%8F%E6%9C%BA%E6%89%B0%E5%8A%A8%E7%82%B9.py
        """
        noise = torch.from_numpy(np.random.normal(0, std, size=point_cloud.shape)).to(point_cloud.device)
        noisy_cloud = point_cloud + noise
        return noisy_cloud

    def gen_init_population(self):
        pass

    def encode(self, points):
        pass

    def extraction(self, points, device, points_per_obj=10, topk=10):
        results = self.model.predict(return_loss=False, rescale=True, **self.data)

        points_all = torch.as_tensor(points, device=device)
        points_xyz = points_all[:, :3]


        scores = results[0]['pts_bbox']['scores_3d']  # (B,)
        boxes = results[0]['pts_bbox']['boxes_3d']    # LiDARInstance3DBoxes

        # top-K selection
        K = min(topk, scores.numel())
        _, topk_inds = torch.topk(scores, K)

        pred_boxes = boxes[topk_inds]

        lidar_boxes = LiDARInstance3DBoxes(
            pred_boxes.tensor[:, :7],
            box_dim=7,
            with_yaw=True
        )

        # point-in-box 
        mask = lidar_boxes.points_in_boxes_all(points_xyz)  # (N, K)

        sampled_points_per_box = []

        for b in range(mask.size(1)):
            box_points = points_all[mask[:, b] > 0]

            n = box_points.size(0)
            if n == 0:
                sampled_points_per_box.append(None)
                continue

            if n >= points_per_obj:
                idx = torch.randperm(n, device=device)[:points_per_obj]
            else:
                idx = torch.randint(0, n, (points_per_obj,), device=device)

            sampled_points_per_box.append(box_points[idx])
        # print("points per boxes: ", sampled_points_per_box)
        return sampled_points_per_box

