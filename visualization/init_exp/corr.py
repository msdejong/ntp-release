import matplotlib.pyplot as plt
import json

input_path = "visualization/init_exp/vizdata/corr.json"
output_path = "visualization/init_exp/vizdata/corr.png"

with open(input_path) as f:
    result_dict = json.load(f)

epochs = len(result_dict['rule_zero'])

x_values = [i for i in range(epochs)]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

rule_color = "darkgreen"
unification_color = "darkblue"

label_font_size = 8
title_font_size = 10

ax1.plot(x_values, result_dict["rule_zero"], label="rule", color=rule_color)
ax1.plot(x_values, result_dict["unification_zero"], label="unification", color=unification_color)

ax2.plot(x_values, result_dict["rule_one"], label="rule", color=rule_color)
ax2.plot(x_values, result_dict["unification_one"], label="unification", color=unification_color)


ax1.legend(loc="right", frameon=False)

ax1.set_xlabel("Epoch", fontsize=label_font_size)
ax1.set_ylabel("Score", fontsize=label_font_size)
ax1.set_aspect(0.4/ax1.get_data_ratio(), share=True)
ax2.set_xlabel("Epoch", fontsize=label_font_size)

ax1.set_title("Relationship not learned", fontsize=title_font_size)
ax2.set_title("Relationship learned", fontsize=title_font_size)

# plt.show()
# ax2.yaxis.set_tick_params(labelbottom=True)
plt.tight_layout()


plt.savefig(output_path, bbox_inches='tight')
