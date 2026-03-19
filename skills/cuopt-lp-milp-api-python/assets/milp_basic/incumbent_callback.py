# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Same MILP as model.py but with a callback to receive incumbent (intermediate) solutions.
MILP only; not for LP.
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
from cuopt.linear_programming.internals import GetSolutionCallback


class IncumbentCallback(GetSolutionCallback):
    def __init__(self, user_data):
        super().__init__()
        self.n_callbacks = 0
        self.user_data = user_data

    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        self.n_callbacks += 1
        sol = (
            solution.tolist()
            if hasattr(solution, "tolist")
            else list(solution)
        )
        cost = float(solution_cost[0])
        print(f"Incumbent {self.n_callbacks}: {sol}, cost: {cost:.2f}")


def main():
    problem = Problem("Incumbent Example")
    x = problem.addVariable(vtype=INTEGER)
    y = problem.addVariable(vtype=INTEGER)
    problem.addConstraint(2 * x + 4 * y >= 230)
    problem.addConstraint(3 * x + 2 * y <= 190)
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    user_data = {"source": "incumbent_callback"}
    settings = SolverSettings()
    settings.set_mip_callback(IncumbentCallback(user_data), user_data)
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)
    problem.solve(settings)

    print(f"Status: {problem.Status.name}, Objective: {problem.ObjValue}")


if __name__ == "__main__":
    main()
