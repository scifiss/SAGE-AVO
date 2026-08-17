# SAGE-AVO Acknowledgments

I am profoundly grateful to **Professor Sergey Fomel** and **Professor Yangkang Chen**, my doctoral supervisors, for supporting and guiding me through every stage of SAGE-AVO. From the earliest understanding of the scientific problem and field data, through method design, experiments, interpretation, validation, and manuscript development, they offered rigorous questions, patient discussion, honest feedback, and steady encouragement. I am especially thankful for the trust they placed in me to explore new ideas and for the generosity with which they shared their time and knowledge. Their mentorship shaped not only this project, but also the way I think, question, and grow as a researcher.

I am sincerely grateful to **Dr. Liuqing Yang** for providing the field data and patiently explaining its acquisition and processing context. Throughout the development of SAGE-AVO, Liuqing gave me many valuable discussions and precious suggestions. His generosity, technical insight, and collegial support were essential in helping this project move forward, and I deeply appreciate the time, knowledge, and encouragement he shared with me.

## HCTNet inspiration and planned comparison

Liuqing’s HCTNet paper was an important source of inspiration during the development of SAGE-AVO. It helped me think more deeply about how convolutional feature extraction and transformer-based context modeling can be used for prestack elastic inversion, and it sharpened the scientific questions that a careful comparison should address.

No HCTNet comparison result is claimed in the current SAGE-AVO release. I plan to develop a faithful, properly attributed reimplementation of HCTNet and compare it with SAGE-AVO under matched data, low-frequency priors, train/validation/test splits, training budgets, checkpoint selection, and evaluation masks. That comparison will be reported only after the reference implementation and experimental protocol are complete.

### Reference

Yang, Liuqing; Fomel, Sergey; Wang, Shoudong; Li, Wenjin; Meng, Jinyu; Li, Chao; and Chen, Yangkang (2025). **HCTNet: Robust prestack seismic inversion using a hybrid convolutional transformer.** *Geophysics*, 90(4), N17-N32. DOI: 10.1190/geo2024-0015.1.

