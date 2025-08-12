#!/bin/bash

# Variables
CHAIN_ID="localnet"
MONIKER0="node0"
MONIKER1="node1"
STAKE_DENOM="stake"
STAKE_AMOUNT="1000000000${STAKE_DENOM}"
NODE0_HOME="/cosmos-multi-node/node0"
NODE1_HOME="/cosmos-multi-node/node1"

# Initialize the nodes
simd init $MONIKER0 --chain-id $CHAIN_ID --home $NODE0_HOME --overwrite
simd init $MONIKER1 --chain-id $CHAIN_ID --home $NODE1_HOME --overwrite

# Add a genesis account to node0
simd keys add validator --keyring-backend test --home $NODE0_HOME
simd add-genesis-account $(simd keys show validator -a --keyring-backend test --home $NODE0_HOME) $STAKE_AMOUNT --home $NODE0_HOME

# Generate gentx for node0
simd gentx validator $STAKE_AMOUNT --chain-id $CHAIN_ID --keyring-backend test --home $NODE0_HOME

# Collect the gentxs
simd collect-gentxs --home $NODE0_HOME

# Validate the genesis file
simd validate-genesis --home $NODE0_HOME

# Copy the genesis file to node1
cp $NODE0_HOME/config/genesis.json $NODE1_HOME/config/genesis.json

# Start the nodes
simd start --home $NODE0_HOME &
simd start --home $NODE1_HOME &
wait -n


