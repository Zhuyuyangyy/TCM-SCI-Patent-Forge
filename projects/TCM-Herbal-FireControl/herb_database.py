"""
herb_database.py - 中药材炮制参数数据库
包含20+种常用中药材的火候控制参数
"""

HERB_DATABASE = {
    # ==================== 根茎类 ====================
    "附子": {
        "name_cn": "附子",
        "name_en": "Aconiti Lateralis Radix",
        "category": "温里药",
        "origin": "毛茛科植物乌头的子根",
        "processing_methods": {
            "生用": {
                "temp_range": (20, 30),
                "temp_target": 25,
                "duration_min": 0,
                "description": "回阳救逆，补火助阳，散寒止痛"
            },
            "制附子(淡附片)": {
                "temp_range": (110, 130),
                "temp_target": 120,
                "duration_min": 240,
                "description": "降低毒性，保留功效"
            },
            "炮附片": {
                "temp_range": (200, 250),
                "temp_target": 230,
                "duration_min": 30,
                "description": "温肾暖脾，用于阳虚泄泻"
            },
            "黑顺片": {
                "temp_range": (130, 150),
                "temp_target": 140,
                "duration_min": 180,
                "description": "炮制后呈黑褐色，毒性降低"
            }
        },
        "fire_level": {"low": 120, "medium": 180, "high": 230},
        "quality_markers": ["乌头碱", "次乌头碱", "新乌头碱"],
        "color_change": {"raw": "灰白色", "processed": "黑褐色"},
        "texture_change": {"raw": "质硬脆", "processed": "质变软，断面有光泽"}
    },

    "白术": {
        "name_cn": "白术",
        "name_en": "Atractylodis Macrocephalae Rhizoma",
        "category": "补气药",
        "origin": "菊科植物白术的根茎",
        "processing_methods": {
            "生白术": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "健脾燥湿，利水消肿"
            },
            "麸炒白术": {
                "temp_range": (150, 180),
                "temp_target": 165,
                "duration_min": 20,
                "description": "缓和燥性，增强健脾作用"
            },
            "土炒白术": {
                "temp_range": (160, 190),
                "temp_target": 175,
                "duration_min": 25,
                "description": "增强补脾止泻作用"
            },
            "焦白术": {
                "temp_range": (180, 220),
                "temp_target": 200,
                "duration_min": 15,
                "description": "用于脾虚泄泻，便血崩漏"
            }
        },
        "fire_level": {"low": 165, "medium": 185, "high": 210},
        "quality_markers": ["白术内酯", "苍术酮", "挥发油"],
        "color_change": {"raw": "黄白色", "processed": "黄棕色至焦褐色"},
        "texture_change": {"raw": "质坚硬", "processed": "表面有焦斑，断面焦黄色"}
    },

    "地黄": {
        "name_cn": "地黄",
        "name_en": "Rehmanniae Radix",
        "category": "清热凉血药",
        "origin": "玄参科植物地黄的块根",
        "processing_methods": {
            "鲜地黄": {
                "temp_range": (20, 30),
                "temp_target": 25,
                "duration_min": 0,
                "description": "清热生津，凉血止血"
            },
            "生地黄": {
                "temp_range": (50, 60),
                "temp_target": 55,
                "duration_min": 480,
                "description": "清热凉血，养阴生津"
            },
            "熟地黄": {
                "temp_range": (100, 110),
                "temp_target": 105,
                "duration_min": 720,
                "description": "补血滋阴，益精填髓"
            },
            "生地炭": {
                "temp_range": (180, 200),
                "temp_target": 190,
                "duration_min": 30,
                "description": "凉血止血，用于出血症"
            },
            "熟地炭": {
                "temp_range": (200, 220),
                "temp_target": 210,
                "duration_min": 25,
                "description": "补血止血，用于血虚出血"
            }
        },
        "fire_level": {"low": 55, "medium": 105, "high": 190},
        "quality_markers": ["梓醇", "地黄苷", "毛蕊花糖苷"],
        "color_change": {"raw": "浅黄色", "processed": "黑褐色"},
        "texture_change": {"raw": "质柔软", "processed": "质变软糯，色黑有光泽"}
    },

    "当归": {
        "name_cn": "当归",
        "name_en": "Angelicae Sinensis Radix",
        "category": "补血药",
        "origin": "伞形科植物当归的根",
        "processing_methods": {
            "生当归": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "补血活血，调经止痛"
            },
            "酒当归": {
                "temp_range": (110, 130),
                "temp_target": 120,
                "duration_min": 15,
                "description": "增强活血作用"
            },
            "土炒当归": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 20,
                "description": "增强补血作用，减少滑肠"
            },
            "当归炭": {
                "temp_range": (180, 200),
                "temp_target": 190,
                "duration_min": 25,
                "description": "止血和血，用于崩漏"
            }
        },
        "fire_level": {"low": 120, "medium": 160, "high": 190},
        "quality_markers": ["藁本内酯", "阿魏酸", "当归多糖"],
        "color_change": {"raw": "黄棕色", "processed": "深褐色"},
        "texture_change": {"raw": "质柔韧", "processed": "微有焦斑，质变酥脆"}
    },

    "甘草": {
        "name_cn": "甘草",
        "name_en": "Glycyrrhizae Radix",
        "category": "补气药",
        "origin": "豆科植物甘草的根和根茎",
        "processing_methods": {
            "生甘草": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "清热解毒，祛痰止咳"
            },
            "蜜炙甘草": {
                "temp_range": (120, 150),
                "temp_target": 135,
                "duration_min": 30,
                "description": "补脾和胃，益气复脉"
            },
            "炒甘草": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 20,
                "description": "健脾益气，用于脾胃虚弱"
            }
        },
        "fire_level": {"low": 135, "medium": 160, "high": 180},
        "quality_markers": ["甘草酸", "甘草苷", "黄酮类"],
        "color_change": {"raw": "红棕色", "processed": "深黄色"},
        "texture_change": {"raw": "质坚硬", "processed": "表面光亮，有蜂蜜光泽"}
    },

    # ==================== 果实种子类 ====================
    "山楂": {
        "name_cn": "山楂",
        "name_en": "Crataegi Fructus",
        "category": "消食药",
        "origin": "蔷薇科植物山里红的成熟果实",
        "processing_methods": {
            "生山楂": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "消食化积，活血散瘀"
            },
            "炒山楂": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 15,
                "description": "消食导滞，用于肉食积滞"
            },
            "焦山楂": {
                "temp_range": (180, 200),
                "temp_target": 190,
                "duration_min": 10,
                "description": "增强消食止泻作用"
            },
            "山楂炭": {
                "temp_range": (220, 250),
                "temp_target": 235,
                "duration_min": 8,
                "description": "收敛止血，用于泄泻崩漏"
            }
        },
        "fire_level": {"low": 160, "medium": 190, "high": 235},
        "quality_markers": ["有机酸", "黄酮类", "三萜类"],
        "color_change": {"raw": "红棕色", "processed": "焦褐色至黑色"},
        "texture_change": {"raw": "质坚硬", "processed": "外焦内酥，略有焦香"}
    },

    "苦杏仁": {
        "name_cn": "苦杏仁",
        "name_en": "Armeniacae Semen Amarum",
        "category": "止咳平喘药",
        "origin": "蔷薇科植物山杏的成熟种子",
        "processing_methods": {
            "生苦杏仁": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "降气止咳，润肠通便（有小毒）"
            },
            "燀苦杏仁": {
                "temp_range": (80, 100),
                "temp_target": 90,
                "duration_min": 10,
                "description": "去皮尖，降低毒性"
            },
            "炒苦杏仁": {
                "temp_range": (120, 140),
                "temp_target": 130,
                "duration_min": 15,
                "description": "温肺散寒，增强止咳功效"
            }
        },
        "fire_level": {"low": 90, "medium": 130, "high": 160},
        "quality_markers": ["苦杏仁苷", "氢氰酸"],
        "color_change": {"raw": "黄棕色", "processed": "黄色至深黄色"},
        "texture_change": {"raw": "种皮薄", "processed": "种皮分离，易脱皮"}
    },

    "槟榔": {
        "name_cn": "槟榔",
        "name_en": "Arecae Semen",
        "category": "驱虫药",
        "origin": "棕榈科植物槟榔的成熟种子",
        "processing_methods": {
            "生槟榔": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "杀虫消积，行气利水"
            },
            "炒槟榔": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 20,
                "description": "增强消积作用"
            },
            "焦槟榔": {
                "temp_range": (180, 200),
                "temp_target": 190,
                "duration_min": 15,
                "description": "消食导滞，用于食积不化"
            }
        },
        "fire_level": {"low": 160, "medium": 190, "high": 210},
        "quality_markers": ["槟榔碱", "鞣质", "脂肪油"],
        "color_change": {"raw": "淡棕色", "processed": "焦褐色"},
        "texture_change": {"raw": "质坚硬", "processed": "断面可见焦斑"}
    },

    # ==================== 皮类 ====================
    "陈皮": {
        "name_cn": "陈皮",
        "name_en": "Citri Reticulatae Pericarpium",
        "category": "理气药",
        "origin": "芸香科植物橘的成熟果皮",
        "processing_methods": {
            "生陈皮": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "理气健脾，燥湿化痰"
            },
            "炒陈皮": {
                "temp_range": (120, 150),
                "temp_target": 135,
                "duration_min": 15,
                "description": "增强理气开胃作用"
            },
            "麸炒陈皮": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 20,
                "description": "缓和燥性，用于脾虚痰湿"
            }
        },
        "fire_level": {"low": 135, "medium": 160, "high": 180},
        "quality_markers": ["橙皮苷", "挥发油", "陈皮素"],
        "color_change": {"raw": "橙红色", "processed": "深褐色"},
        "texture_change": {"raw": "质柔软", "processed": "微有光泽，质变脆"}
    },

    "厚朴": {
        "name_cn": "厚朴",
        "name_en": "Magnoliae Officinalis Cortex",
        "category": "化湿药",
        "origin": "木兰科植物厚朴的干皮",
        "processing_methods": {
            "生厚朴": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "燥湿消痰，下气除满"
            },
            "姜厚朴": {
                "temp_range": (130, 150),
                "temp_target": 140,
                "duration_min": 25,
                "description": "消除对咽喉的刺激，和胃止呕"
            }
        },
        "fire_level": {"low": 140, "medium": 160, "high": 180},
        "quality_markers": ["厚朴酚", "和厚朴酚", "挥发油"],
        "color_change": {"raw": "紫棕色", "processed": "深紫色"},
        "texture_change": {"raw": "质坚硬", "processed": "内有姜汁渗出"}
    },

    # ==================== 全草类 ====================
    "麻黄": {
        "name_cn": "麻黄",
        "name_en": "Ephedrae Herba",
        "category": "解表药",
        "origin": "麻黄科植物草麻黄的草质茎",
        "processing_methods": {
            "生麻黄": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "发汗解表，宣肺平喘"
            },
            "蜜麻黄": {
                "temp_range": (100, 120),
                "temp_target": 110,
                "duration_min": 20,
                "description": "润肺止咳，用于表证已解"
            },
            "麻黄绒": {
                "temp_range": (80, 100),
                "temp_target": 90,
                "duration_min": 15,
                "description": "缓和发汗作用，用于老人幼儿"
            }
        },
        "fire_level": {"low": 90, "medium": 110, "high": 140},
        "quality_markers": ["麻黄碱", "伪麻黄碱", "挥发油"],
        "color_change": {"raw": "黄绿色", "processed": "深绿色"},
        "texture_change": {"raw": "质脆", "processed": "微有黏性，有蜂蜜光泽"}
    },

    # ==================== 矿物类 ====================
    "石膏": {
        "name_cn": "石膏",
        "name_en": "Gypsum Fibrosum",
        "category": "清热泻火药",
        "origin": "硫酸盐类矿物石膏的矿石",
        "processing_methods": {
            "生石膏": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "清热泻火，除烦止渴"
            },
            "煅石膏": {
                "temp_range": (400, 500),
                "temp_target": 450,
                "duration_min": 60,
                "description": "收敛生肌，用于疮疡溃不收口"
            }
        },
        "fire_level": {"low": 200, "medium": 350, "high": 450},
        "quality_markers": ["硫酸钙", "杂质"],
        "color_change": {"raw": "白色", "processed": "灰白色"},
        "texture_change": {"raw": "质重", "processed": "质松脆，易粉碎"}
    },

    "赭石": {
        "name_cn": "赭石",
        "name_en": "Haematitum",
        "category": "平肝潜阳药",
        "origin": "氧化物类矿物赤铁矿的矿石",
        "processing_methods": {
            "生赭石": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "平肝潜阳，重镇降逆"
            },
            "煅赭石": {
                "temp_range": (300, 400),
                "temp_target": 350,
                "duration_min": 45,
                "description": "降低苦寒之性，增强平肝止血作用"
            }
        },
        "fire_level": {"low": 250, "medium": 350, "high": 450},
        "quality_markers": ["氧化铁", "黏土"],
        "color_change": {"raw": "暗棕红色", "processed": "赭红色"},
        "texture_change": {"raw": "质坚硬", "processed": "质变酥脆，易粉碎"}
    },

    # ==================== 动物类 ====================
    "地龙": {
        "name_cn": "地龙",
        "name_en": "Pheretima",
        "category": "平肝息风药",
        "origin": "钜蚓科动物参环毛蚓的全体",
        "processing_methods": {
            "生地龙": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "清热定惊，通络平喘"
            },
            "酒地龙": {
                "temp_range": (120, 140),
                "temp_target": 130,
                "duration_min": 15,
                "description": "增强通络作用，便于服用"
            },
            "炒地龙": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 10,
                "description": "矫臭矫味，用于惊痫抽搐"
            }
        },
        "fire_level": {"low": 130, "medium": 160, "high": 180},
        "quality_markers": ["蚯蚓素", "氨基酸", "微量元素"],
        "color_change": {"raw": "灰棕色", "processed": "焦黄色"},
        "texture_change": {"raw": "体轻", "processed": "质脆，有焦香气"}
    },

    "僵蚕": {
        "name_cn": "僵蚕",
        "name_en": "Bombyx Batryticatus",
        "category": "平肝息风药",
        "origin": "蚕蛾科昆虫家蚕的幼虫感染白僵菌的干燥体",
        "processing_methods": {
            "生僵蚕": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "息风止痉，祛风止痛"
            },
            "麸炒僵蚕": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 15,
                "description": "矫臭矫味，增强健脾作用"
            },
            "炒僵蚕": {
                "temp_range": (160, 180),
                "temp_target": 170,
                "duration_min": 12,
                "description": "用于惊痫抽搐"
            }
        },
        "fire_level": {"low": 160, "medium": 170, "high": 190},
        "quality_markers": ["蛋白质", "脂肪", "草酸铵"],
        "color_change": {"raw": "灰白色", "processed": "焦黄色"},
        "texture_change": {"raw": "质硬而脆", "processed": "表面有麸皮附着"}
    },

    # ==================== 其他类 ====================
    "延胡索": {
        "name_cn": "延胡索",
        "name_en": "Corydalis Rhizoma",
        "category": "活血止痛药",
        "origin": "罂粟科植物延胡索的块茎",
        "processing_methods": {
            "生延胡索": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "活血行气止痛"
            },
            "醋延胡索": {
                "temp_range": (120, 140),
                "temp_target": 130,
                "duration_min": 20,
                "description": "增强止痛作用，醋制引药入肝"
            },
            "酒延胡索": {
                "temp_range": (110, 130),
                "temp_target": 120,
                "duration_min": 15,
                "description": "用于跌打损伤"
            }
        },
        "fire_level": {"low": 120, "medium": 130, "high": 160},
        "quality_markers": ["延胡索乙素", "去氢延胡索甲素"],
        "color_change": {"raw": "黄色", "processed": "深黄色至黄褐色"},
        "texture_change": {"raw": "质坚硬", "processed": "表面有醋迹"}
    },

    "何首乌": {
        "name_cn": "何首乌",
        "name_en": "Polygoni Multiflori Radix",
        "category": "补血药",
        "origin": "蓼科植物何首乌的块根",
        "processing_methods": {
            "生何首乌": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "解毒，截疟，润肠通便"
            },
            "制何首乌": {
                "temp_range": (100, 110),
                "temp_target": 105,
                "duration_min": 480,
                "description": "补肝肾，益精血，乌须发"
            },
            "黑豆汁蒸何首乌": {
                "temp_range": (100, 110),
                "temp_target": 105,
                "duration_min": 600,
                "description": "增强补益作用"
            }
        },
        "fire_level": {"low": 55, "medium": 105, "high": 160},
        "quality_markers": ["二苯乙烯苷", "卵磷脂", "蒽醌类"],
        "color_change": {"raw": "红棕色", "processed": "黑褐色"},
        "texture_change": {"raw": "质坚硬", "processed": "质柔软，有光泽"}
    },

    "大黄": {
        "name_cn": "大黄",
        "name_en": "Rhei Radix et Rhizoma",
        "category": "泻下药",
        "origin": "蓼科植物掌叶大黄的根和根茎",
        "processing_methods": {
            "生大黄": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "泻下攻积，清热泻火"
            },
            "熟大黄": {
                "temp_range": (100, 110),
                "temp_target": 105,
                "duration_min": 180,
                "description": "缓和泻下作用，增强活血作用"
            },
            "酒大黄": {
                "temp_range": (120, 140),
                "temp_target": 130,
                "duration_min": 15,
                "description": "引药上行，清上焦血分热毒"
            },
            "大黄炭": {
                "temp_range": (200, 220),
                "temp_target": 210,
                "duration_min": 20,
                "description": "凉血止血，用于血热出血"
            }
        },
        "fire_level": {"low": 105, "medium": 150, "high": 210},
        "quality_markers": ["蒽醌苷", "鞣质", "二苯乙烯苷"],
        "color_change": {"raw": "黄棕色", "processed": "焦黑色"},
        "texture_change": {"raw": "质坚实", "processed": "外焦内黑，有炭香"}
    },

    "黄芪": {
        "name_cn": "黄芪",
        "name_en": "Astragali Radix",
        "category": "补气药",
        "origin": "豆科植物蒙古黄芪的根",
        "processing_methods": {
            "生黄芪": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "补气升阳，固表止汗"
            },
            "蜜炙黄芪": {
                "temp_range": (120, 150),
                "temp_target": 135,
                "duration_min": 30,
                "description": "增强补中益气作用，用于气血两虚"
            },
            "炒黄芪": {
                "temp_range": (150, 170),
                "temp_target": 160,
                "duration_min": 20,
                "description": "健脾和胃，用于脾虚食少"
            }
        },
        "fire_level": {"low": 135, "medium": 160, "high": 180},
        "quality_markers": ["黄芪甲苷", "黄芪多糖", "毛蕊异黄酮"],
        "color_change": {"raw": "淡黄棕色", "processed": "深黄色"},
        "texture_change": {"raw": "质硬而韧", "processed": "表面有蜂蜜光泽"}
    },

    "黄连": {
        "name_cn": "黄连",
        "name_en": "Coptidis Rhizoma",
        "category": "清热燥湿药",
        "origin": "毛茛科植物黄连的根茎",
        "processing_methods": {
            "生黄连": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "清热燥湿，泻火解毒"
            },
            "酒黄连": {
                "temp_range": (120, 140),
                "temp_target": 130,
                "duration_min": 15,
                "description": "引药上行，清上焦火热"
            },
            "姜黄连": {
                "temp_range": (130, 150),
                "temp_target": 140,
                "duration_min": 20,
                "description": "清胃和胃止呕，用于胃热呕吐"
            },
            "萸黄连": {
                "temp_range": (130, 150),
                "temp_target": 140,
                "duration_min": 20,
                "description": "舒肝和胃止呕，用于肝胃不和"
            }
        },
        "fire_level": {"low": 130, "medium": 140, "high": 160},
        "quality_markers": ["小檗碱", "黄连碱", "巴马汀"],
        "color_change": {"raw": "金黄色", "processed": "深黄色"},
        "texture_change": {"raw": "质坚硬", "processed": "微有姜汁或吴萸汁痕迹"}
    },

    "杜仲": {
        "name_cn": "杜仲",
        "name_en": "Eucommiae Cortex",
        "category": "补阳药",
        "origin": "杜仲科植物杜仲的树皮",
        "processing_methods": {
            "生杜仲": {
                "temp_range": (20, 35),
                "temp_target": 28,
                "duration_min": 0,
                "description": "补肝肾，强筋骨，安胎"
            },
            "盐杜仲": {
                "temp_range": (150, 180),
                "temp_target": 165,
                "duration_min": 25,
                "description": "引药入肾，增强补肾强腰作用"
            },
            "炒杜仲": {
                "temp_range": (160, 190),
                "temp_target": 175,
                "duration_min": 20,
                "description": "温肾助阳，用于阳虚腰痛"
            }
        },
        "fire_level": {"low": 165, "medium": 175, "high": 200},
        "quality_markers": ["松脂醇二葡萄糖苷", "桃叶珊瑚苷", "黄酮类"],
        "color_change": {"raw": "灰棕色", "processed": "焦黄色至黑棕色"},
        "texture_change": {"raw": "质脆", "processed": "折断有胶丝相连"}
    }
}


def get_herb_info(herb_name: str) -> dict:
    """获取药材信息"""
    return HERB_DATABASE.get(herb_name, None)


def get_processing_methods(herb_name: str) -> list:
    """获取药材的炮制方法列表"""
    herb = HERB_DATABASE.get(herb_name)
    if herb:
        return list(herb.get("processing_methods", {}).keys())
    return []


def get_fire_parameters(herb_name: str, method: str) -> dict:
    """获取特定炮制方法的火候参数"""
    herb = HERB_DATABASE.get(herb_name)
    if herb and method in herb.get("processing_methods", {}):
        return herb["processing_methods"][method]
    return None


def list_all_herbs() -> list:
    """列出所有药材名称"""
    return list(HERB_DATABASE.keys())


def get_category_herbs(category: str) -> list:
    """按类别获取药材"""
    return [name for name, herb in HERB_DATABASE.items() 
            if herb.get("category") == category]
