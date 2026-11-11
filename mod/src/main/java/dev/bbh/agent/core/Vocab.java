package dev.bbh.agent.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Inventory/goal vocabulary written by the training notebook (vocab.json). Goal id = index + 1, 0 = no goal. */
public final class Vocab {
    public final List<String> names;
    private final Map<String, Integer> index = new HashMap<>();

    public Vocab(List<String> names) {
        this.names = List.copyOf(names);
        for (int i = 0; i < names.size(); i++) index.put(names.get(i), i);
    }

    public static Vocab fromJson(String json) {
        List<String> l = new ArrayList<>();
        for (Object o : MiniJson.array(MiniJson.parse(json))) l.add((String) o);
        return new Vocab(l);
    }

    /** Index in the state vector, or -1 if the item is not in the vocabulary. */
    public int indexOf(String canonName) {
        return index.getOrDefault(canonName, -1);
    }

    /** Goal id for the policy: 0 when unknown. Accepts raw registry names such as "minecraft:oak_log". */
    public int goalId(String rawName) {
        int i = indexOf(Canon.item(rawName));
        return i < 0 ? 0 : i + 1;
    }
}
