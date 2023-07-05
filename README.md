# DouNPC

This work is based on DouZero. All extra files will end with "_npc".

## Extra Files

test_npc.py

train_npc.py

douzero/env/game_npc.py

douzero/evaluation/simulation_npc.py

- mp_record(card_play_data_list, card_play_model_path_dict, q)




## Original Work: DouZero

Zha, Daochen et al. “DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning.” ICML (2021).

```bibtex
@InProceedings{pmlr-v139-zha21a,
  title = 	 {DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning},
  author =       {Zha, Daochen and Xie, Jingru and Ma, Wenye and Zhang, Sheng and Lian, Xiangru and Hu, Xia and Liu, Ji},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12333--12344},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zha21a/zha21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/zha21a.html},
  abstract = 	 {Games are abstractions of the real world, where artificial agents learn to compete and cooperate with other agents. While significant achievements have been made in various perfect- and imperfect-information games, DouDizhu (a.k.a. Fighting the Landlord), a three-player card game, is still unsolved. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. Unfortunately, modern reinforcement learning algorithms mainly focus on simple and small action spaces, and not surprisingly, are shown not to make satisfactory progress in DouDizhu. In this work, we propose a conceptually simple yet effective DouDizhu AI system, namely DouZero, which enhances traditional Monte-Carlo methods with deep neural networks, action encoding, and parallel actors. Starting from scratch in a single server with four GPUs, DouZero outperformed all the existing DouDizhu AI programs in days of training and was ranked the first in the Botzone leaderboard among 344 AI agents. Through building DouZero, we show that classic Monte-Carlo methods can be made to deliver strong results in a hard domain with a complex action space. The code and an online demo are released at https://github.com/kwai/DouZero with the hope that this insight could motivate future work.}
}
```

