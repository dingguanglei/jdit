# coding=utf-8
from abc import abstractmethod
from multiprocessing import Pool
from types import FunctionType

class SupParallelTrainer(object):
    """ Training parallel.

    .. attr::`default_params` is the default params.

    .. attr::`unfixed_params_list` is the different params.


    """
    def __init__(self, default_params:dict, unfixed_params_list:list):
        """

        :param default_params: a ``dict()`` like {param:v1, param:v2 ...}
        :param unfixed_params_list:  a ``list`` like [{param:v1, param:v2}, {param:v1, param:v2}, ...].
        You must set the value of `task_id` and `gpu_ids_abs`, like {'task_id': 1}. {'gpu_ids_abs': [0,1]},
        regardless in ``default_params`` or ``unfixed_params_list``.

        .. note ::

            You must set the value of `task_id` and `gpu_ids_abs`, like {'task_id': 1}. {'gpu_ids_abs': [0,1]}

        """
        self.default_params = default_params
        # {params:123}
        self.unfixed_params_list = unfixed_params_list  #
        # self.unfixed_params_list =[{'depth':12},{'depth':18},{'depth':24},{'depth':26}]

        self.candidate_params_list = self._build_candidate_params(default_params, unfixed_params_list)
        # [{params1..},{params2...}]
        self.parallel_plans = self._distribute_task_on_devices(self.candidate_params_list)
        # self.parallel_plans = {(task_id):[{param1},{param2}]}

    @abstractmethod
    def build_task_trainer(self, params:dict):
        """You need to write this method to build your own ``Trainer``.

        This will run in a certain subprocess.
        The keys of ``params`` are compatible with ``dataset`` , ``Model`` , ``Optimizer`` and ``Trainer`` .
        You can see parameters in the following example.

        These two parameters are special.

        * ``params["logdir"]``   controls the log directory.
        * ``params["gpu_ids_abs"]`` controls the running devices.

        You should return a ``Trainer`` when you finish you building.

        :param params: parameters dictionary.
        :return: Trainer

        Example::

            # Using ``params['key']`` to build your Trainer.
            logdir = params["logdir"] # necessary!
            gpu_ids_abs = params["gpu_ids_abs"] # necessary!
            use_benchmark = params["use_benchmark"]
            data_root = params["data_root"]
            batch_shape = params["batch_shape"]
            opt_name = params["opt_name"]
            lr = params["lr"]
            lr_decay = params["lr_decay"]
            lr_minimum = params["lr_minimum"]
            weight_decay = params["weight_decay"]
            momentum = params["momentum"]
            betas = params["betas"]
            init_method = params["init_method"]
            depth = params["depth"]
            mid_channels = params["mid_channels"]
            nepochs = params["nepochs"]

            torch.backends.cudnn.benchmark = use_benchmark
            mnist = FashionMNIST(root=data_root, batch_shape=batch_shape)
            T_net = Model(Tresnet18(depth=depth, mid_channels=mid_channels), gpu_ids_abs=gpu_ids_abs,
                          init_method=init_method)
            opt = Optimizer(T_net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name,
                            lr_minimum=lr_minimum)
            Trainer = FashingClassTrainer(logdir, nepochs, gpu_ids_abs, T_net, opt, mnist)
            # You must return a Trainer!
            return Trainer

        """

        pass

    def train(self, max_processes=4):
        """start parallel task

        To start the parallel task that were saved in  ``self.parallel_plans`` dictionary.
        :param max_processes: A max amount of processes for setting ``Pool(processes = ?)`` method.
        """
        # print("Main process ID: %d" % os.getpid())
        print('Waiting for all subprocesses done...\n%s' % ('=' * 36))
        p = Pool(max_processes)
        # {(gpu_id1):[{param1}, {param2}], (gpu_id2):[{param1}, {param2}]}
        for position, parallel_plan in enumerate(self.parallel_plans.items()):
            # task_id, candidate_params = parallel_plan
            p.apply_async(self._start_train, (parallel_plan, position),
                          callback=self.finish, error_callback=self.error)
        # p.map_async(self._start_train, self.parallel_plans.items(), callback=self.finish, error_callback=self.error)

        p.close()
        p.join()
        print('All subprocesses done.')

    def _start_train(self, parallel_plan:tuple, position:int):
        task_id, candidate_params = parallel_plan
        # task_id, candidate_params = parallel_planChild process ID:
        nums_tasks = len(candidate_params)
        # print("Task %s child process ID: %d" % (task_id, os.getpid()))
        for index, params in enumerate(candidate_params):
            tag = "CPU" if not params["gpu_ids_abs"] else "GPU%s" % str(params["gpu_ids_abs"])
            process_bar_header = ">>>T%d:(%d/%d)|%s" % (task_id, index, nums_tasks, tag)
            trainer = self.build_task_trainer(params)
            trainer.train(process_bar_header=process_bar_header, process_bar_position=position, subbar_disable=True)
            # print("<<< finish Task %d|%s" % (index, str(task_id)))

    def _distribute_task_on_devices(self, candidate_params_list:list):
        for params in candidate_params_list:
            assert "gpu_ids_abs" in params and "task_id" in params, "You must pass params `gpu_ids_abs` to set device"
            assert "task_id" in params, "You must pass params `task_id` to set a task ID"
        tasks_plan = dict({})  # (task_id):[t3],(task_id):[t1,t2]
        for candidate_params in candidate_params_list:
            task_id = candidate_params["task_id"]
            if task_id in tasks_plan:
                # if task_id have been used, append to the former tasks.
                tasks_plan[task_id].append(candidate_params)
            else:
                # if task_id  have not been used, create a new task list.
                tasks_plan[task_id] = [candidate_params]
        # trainers_plan = list(gpu_used_plan.values)  # [[t1,t2],[t3]...]
        return tasks_plan

    def _build_candidate_params(self, default_params:dict, unfixed_params_list:list):
        final_unfixed_params_list = self._add_logdirs_to_unfixed_params(unfixed_params_list)
        total_params = []
        import copy
        for unfixedparams_dict in final_unfixed_params_list:
            params = copy.deepcopy(default_params)
            for key, value in unfixedparams_dict.items():
                params[key] = value
            total_params.append(copy.deepcopy(params))
        return total_params

    def _add_logdirs_to_unfixed_params(self, unfixed_params_list:list):
        import copy
        final_unfixed_params_list = copy.deepcopy(unfixed_params_list)
        use_auto_logdir = not "logdir" in unfixed_params_list[0]
        if use_auto_logdir:
            print("Build log directories automatically!")
            for index, params_dict in enumerate(unfixed_params_list):  # [dict(),dict()]
                logdir_name = []
                for key, value in params_dict.items():  # params_dict = {p1:1, p2:2}
                    if key == "task_id":
                        continue
                    if key == 'gpu_ids_abs':
                        key = 'gpu'
                    param_name = "=".join([str(key), str(value)])
                    logdir_name.append(param_name)
                    final_unfixed_params_list[index]["logdir"] = "plog/" + ",".join(logdir_name)
        else:
            for index, params_dict in enumerate(unfixed_params_list):  # [dict(),dict()]
                final_unfixed_params_list[index]["logdir"] = self._convert_to_dirname(
                        unfixed_params_list[index]["logdir"])

        print("logdir names are:\n\t%s" % "\n\t".join([params["logdir"] for params in final_unfixed_params_list]))

        return final_unfixed_params_list  # [dir1, dir2, dir3]

    def _convert_to_dirname(self, item:str):
        dir_name = item.strip()
        replace_dict = {"*": "",
                        ">": "greater",
                        "<": "smaller",
                        "|": "-",
                        ":": "%",
                        "?": "$",
                        "/": "_",
                        "\\": "_",
                        }
        for key, value in replace_dict.items():
            dir_name = str(dir_name).replace(key, value)
            if len(dir_name) > 50:
                import warnings
                warnings.warn("the length of `dir_name`(%d) is greater than 50."
                              "It will be cut to `dir_name[0:50]`" % len(dir_name))
                dir_name = dir_name[0:50]
        return dir_name

    def finish(self, msg):
        """When a subprocess finished, it will be called.

        You can rewrite this method for your purpose.
        :param msg: fin
        """

        # print("%s finished!" % os.getpid(), msg)
        pass

    def error(self, msg):
        """When a subprocess failed, it will be called.

        You can rewrite this method for your purpose.
        :param msg: error massage
        """
        print(msg)