# Deleted backup manifest

Repair backups removed from `data/` on **2026-09-15** to reclaim ~527 MB. This file is the
forensic trail that replaces them: every row records what the backup held and what the
current file holds, so a later claim about a repair is checkable without keeping the bytes.

**Why they were deletable.** In both cases the repair is DOCUMENTED, its script is in the
repo, and the REPAIRED state is published: `whr778/cc_news_haiku45` and
`whr778/synthetic_haiku45_5k` on the Hub carry the repaired files and their cards carry the
repair history. What the backups held was the PRE-repair state -- the contaminated version,
which nothing should ever train on.

**What was genuinely forfeited, stated plainly.** These were the only copies of the
pre-repair data. The repairs are not invertible from the repaired file: you cannot
un-remove a seeded negative or un-stamp metadata and recover the original bytes. If a
future question needs the exact pre-repair corpus, it must be REGENERATED from source by
re-running the converter, not restored.

**Also deleted:** three `casualty_loc_split_card.*.precardinality` files (38 MB), verified
byte-identical by sha256 to the `casualty_loc_split` control corpus, which is still present.
Those were pure duplicates and carry no manifest row.

Note the line counts are UNCHANGED in every row: both repairs were label-level, not
record-level. Nothing was dropped from any corpus.

| corpus file | backup | lines before | lines after | bytes before | bytes after | delta |
|---|---|--:|--:|--:|--:|--:|
| `cc_news_haiku45.test.jsonl` | .prerepair | 2,043 | 2,043 | 7,871,456 | 7,866,827 | -4,629 B |
| `cc_news_haiku45.train.jsonl` | .prerepair | 15,839 | 15,839 | 62,291,006 | 62,255,767 | -35,239 B |
| `cc_news_haiku45.val.jsonl` | .prerepair | 2,075 | 2,075 | 8,144,953 | 8,140,225 | -4,728 B |
| `synthetic_haiku45_5k.test.jsonl` | .prerepair | 483 | 483 | 1,569,172 | 1,568,049 | -1,123 B |
| `synthetic_haiku45_5k.train.jsonl` | .prerepair | 4,018 | 4,018 | 13,000,225 | 12,991,800 | -8,425 B |
| `synthetic_haiku45_5k.val.jsonl` | .prerepair | 496 | 496 | 1,607,972 | 1,606,761 | -1,211 B |
| `casualty_docee.test.jsonl` | .bak | 1,906 | 1,906 | 2,345,074 | 2,499,158 | +154,084 B |
| `casualty_docee.train.jsonl` | .bak | 13,358 | 13,358 | 15,975,459 | 17,052,887 | +1,077,428 B |
| `casualty_docee.val.jsonl` | .bak | 1,682 | 1,682 | 1,971,397 | 2,107,224 | +135,827 B |
| `casualty_ft.test.jsonl` | .bak | 1,038 | 1,038 | 442,700 | 526,160 | +83,460 B |
| `casualty_ft.train.jsonl` | .bak | 29,198 | 29,198 | 12,572,192 | 14,922,445 | +2,350,253 B |
| `casualty_ft.val.jsonl` | .bak | 1,303 | 1,303 | 563,671 | 668,489 | +104,818 B |
| `casualty_multi.test.jsonl` | .bak | 1,023 | 1,023 | 1,044,546 | 1,126,497 | +81,951 B |
| `casualty_multi.train.jsonl` | .bak | 29,030 | 29,030 | 29,408,697 | 31,735,958 | +2,327,261 B |
| `casualty_multi.val.jsonl` | .bak | 1,297 | 1,297 | 1,313,982 | 1,417,909 | +103,927 B |
| `casualty_multi_loc.train.jsonl` | .bak | 29,324 | 29,324 | 37,323,650 | 39,720,484 | +2,396,834 B |
| `cc_news_haiku45.test.jsonl` | .bak | 2,043 | 2,043 | 7,860,053 | 7,866,827 | +6,774 B |
| `cc_news_haiku45.train.jsonl` | .bak | 15,839 | 15,839 | 62,208,261 | 62,255,767 | +47,506 B |
| `cc_news_haiku45.val.jsonl` | .bak | 2,075 | 2,075 | 8,134,010 | 8,140,225 | +6,215 B |
| `mix_natural.train.jsonl` | .bak | 84,279 | 84,279 | 117,152,879 | 117,381,901 | +229,022 B |
| `mix_natural.val.jsonl` | .bak | 1,719 | 1,719 | 2,385,568 | 2,390,487 | +4,919 B |
| `synthetic_haiku45_5k.test.jsonl` | .bak | 483 | 483 | 1,559,470 | 1,568,049 | +8,579 B |
| `synthetic_haiku45_5k.train.jsonl` | .bak | 4,018 | 4,018 | 12,915,430 | 12,991,800 | +76,370 B |
| `synthetic_haiku45_5k.val.jsonl` | .bak | 496 | 496 | 1,598,310 | 1,606,761 | +8,451 B |
| `synthetic_haiku45_5k_coerced.test.jsonl` | .bak | 484 | 484 | 1,405,237 | 1,416,367 | +11,130 B |
| `synthetic_haiku45_5k_coerced.train.jsonl` | .bak | 4,020 | 4,020 | 11,584,835 | 11,673,942 | +89,107 B |
| `synthetic_haiku45_5k_coerced.val.jsonl` | .bak | 496 | 496 | 1,422,521 | 1,435,123 | +12,602 B |
| `synthetic_sonnet5_1k.test.jsonl` | .bak | 194 | 194 | 923,492 | 945,331 | +21,839 B |
| `synthetic_sonnet5_1k.train.jsonl` | .bak | 1,497 | 1,497 | 7,196,500 | 7,360,769 | +164,269 B |
| `synthetic_sonnet5_1k.val.jsonl` | .bak | 191 | 191 | 921,709 | 943,229 | +21,520 B |
| `warmstart_mix.train.jsonl` | .bak | 83,880 | 83,880 | 114,489,309 | 116,836,638 | +2,347,329 B |
| `warmstart_mix.val.jsonl` | .bak | 1,718 | 1,718 | 2,369,872 | 2,419,128 | +49,256 B |

sha256 of each deleted backup, so a later claim about its contents is checkable:

```
4d14d5f3a8cc1157783b9f7e33b62e9e58c0e86db364be48450db4331d0a9af4  cc_news_haiku45.test.jsonl.prerepair
2c7463deeaa09ff85049133743633cb64ce017a32e7ab04a35300659675e1eda  cc_news_haiku45.train.jsonl.prerepair
d29d91329531ee49496074541c54973c2b159838c1ad3ec1ace2fcbb46049413  cc_news_haiku45.val.jsonl.prerepair
764df5300327024ea6dcc6491e129f544f58befc9a9e73fedb8e89dca55f9aad  synthetic_haiku45_5k.test.jsonl.prerepair
1695c7e54e259fe590b5300283b68d2762b78c2e2db5ffdf4998ff43f55a8e8c  synthetic_haiku45_5k.train.jsonl.prerepair
91268b654bcf95370ff515ca07197467268908a3387fa0acf556d5553165559d  synthetic_haiku45_5k.val.jsonl.prerepair
b9b35a7a17a24152473f184ad9349b2c6f58efd4bf7e76486b8b0046fe654461  casualty_docee.test.jsonl.bak
58d7b682df2712a49ee025fde57466dee7c2dde60b3253aa0c5ecb889e26fd37  casualty_docee.train.jsonl.bak
45c21e308f4d826ea396d5268da869a20275e976261286720a9f5f39f6bc669d  casualty_docee.val.jsonl.bak
302c26218a57c93330e2147ce43cf92d394cb4af6682055424d87844733291e9  casualty_ft.test.jsonl.bak
2bb9499b9a4ea3de7f321c4d7b63eb015e48d377f6166c087456f2b0babc3e41  casualty_ft.train.jsonl.bak
0f93260fdcce6cb9ccca8802e7760cf4cfe4f0ebc036ac74a620008e6d59c8b2  casualty_ft.val.jsonl.bak
e9327e6615f44e64d5432ace5dd1c02305ab5f7be0292bcd0059134c1d1b754d  casualty_multi.test.jsonl.bak
2a47191a63f6b03d237e99a43c36b3c23d01b52c78725d62334c378220848fbd  casualty_multi.train.jsonl.bak
73c16c8f1cb7080d1921e6d495064730ba1ce56059ca28978c99738554a91ed2  casualty_multi.val.jsonl.bak
d92221a1924d3602c6933f8bd3170947aa110ab6ac722cb0a443c14c0d8a1b42  casualty_multi_loc.train.jsonl.bak
2fe28bec28e57ae9e4c61baf4df1c5cf05277de10f616c54bdfb262bf7a30334  cc_news_haiku45.test.jsonl.bak
90168f92d8aba4531762d0c3d07a50e15426782a55413a5bb917b608e6e7ccbd  cc_news_haiku45.train.jsonl.bak
7c666c87b6ae43d86515e67d544546edc2d116545cc94e96e45a61a2c2e43a96  cc_news_haiku45.val.jsonl.bak
3c5637069aa6f315cd69cd99ec15fd038e4ba636cdb870fafea2efc6cce25709  mix_natural.train.jsonl.bak
d9353d6c1ab0dea6d5c4fb489a8cd4b12cbcf00386db9f823e066ba27f28bfed  mix_natural.val.jsonl.bak
c259c48f8e6d3285225353cb137eb7d115e6a71a0ccfad47dad2ef56c4f6f291  synthetic_haiku45_5k.test.jsonl.bak
f81f426067cbefb2e86d6497121fe508516f0db6ad510506f6b935fd38544883  synthetic_haiku45_5k.train.jsonl.bak
23ff1786e0fb5f814911b54f154ef90d8bfaace6cb9df43bcf4860100a450ab5  synthetic_haiku45_5k.val.jsonl.bak
b753eab17006cb903dffaa7d93f27bedcf763d9b6012b8aa7678d24441987c7b  synthetic_haiku45_5k_coerced.test.jsonl.bak
02381ccbdf29627d32f0ca4e902a01e0773aa7ff1b79b4c80819628462773f02  synthetic_haiku45_5k_coerced.train.jsonl.bak
d1ab1570dd6761aa8cf0f1db1500bcc238678c8621cc95b5779ddce41724d879  synthetic_haiku45_5k_coerced.val.jsonl.bak
c7a896f18168f2e09c212de6ad0d3eb295d4d07acfe5f76883e8e10c9daff907  synthetic_sonnet5_1k.test.jsonl.bak
880fbfef6785c0a343599e2dfed723f9742ab5721400260d198deb1ae9818ce2  synthetic_sonnet5_1k.train.jsonl.bak
6b0d69c40883c9e4c824688efccb2901f9e02cf3b3abc1ff5983d6dd481ae52a  synthetic_sonnet5_1k.val.jsonl.bak
11444153ed8c290736f9a1727172148d9fbb7eeb3df3ff7cb3c1b7d311e3de30  warmstart_mix.train.jsonl.bak
16b5963fca1a62af1a89f1cc3411121a0652b582d8711f0669491befc3ad5d1d  warmstart_mix.val.jsonl.bak
```
