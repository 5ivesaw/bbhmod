package dev.bbh.agent.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Which blocks Baritone should mine to obtain a goal item. Items without an entry cannot be delegated. */
public final class GoalBlocks {
    private final Map<String, String[]> table = new HashMap<>();

    public static GoalBlocks fromJson(String json) {
        GoalBlocks g = new GoalBlocks();
        for (Map.Entry<String, Object> e : MiniJson.object(MiniJson.parse(json)).entrySet()) {
            List<Object> l = MiniJson.array(e.getValue());
            String[] names = new String[l.size()];
            for (int i = 0; i < names.length; i++) names[i] = (String) l.get(i);
            g.table.put(e.getKey(), names);
        }
        return g;
    }

    public static GoalBlocks empty() {
        return new GoalBlocks();
    }

    /** @return block names for the canonical goal item, or null if it cannot be mined directly */
    public String[] blocksFor(String canonItem) {
        return table.get(canonItem);
    }
}
