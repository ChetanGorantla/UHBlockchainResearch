###############################################################################
###                               Localnet.ps1                              ###
###############################################################################
###                                                                         ###
### How to use?                                                             ###
###                                                                         ###
### `.\localnet.ps1`         runs the entire script including clearing old  ###
### data and re running the default setup and then starts the chain.        ###
###                                                                         ###
### `.\localnet.ps1 -Run`    skips the cleanup & setup and directly starts  ###
###  the chain from existing data and continues from old height.            ###
###############################################################################

Param(
    # If specified, the localnet setup is bypassed and the chain is directly run
    [Parameter(Mandatory=$false)]
    [Switch]$Run
)

###############################################################################
###                               Script Vars                               ###
###############################################################################
$CHAIN_ID = "localnet"
$CHAIN_DIR = "data/localnet"
$NODE0_DIR = "./node0"
$NODE1_DIR = "./node1"
$BINARY = "simd"
$BINARY_PATH = Join-Path -Path "/usr/local/bin" -ChildPath $BINARY
$CHAIN_DATA = Join-Path -Path $NODE0_DIR -ChildPath $CHAIN_DIR

## If the param is set, chain setup is not run
if ($Run -eq $false) {

###############################################################################
###                         Application Data Vars                           ###
###############################################################################
$MNEMONIC_1 = "guard cream sadness conduct invite crumble clock pudding hole grit liar hotel maid produce squeeze return argue turtle know drive eight casino maze host"
$MNEMONIC_2 = "friend excite rough reopen cover wheel spoon convince island path clean monkey play snow number walnut pull lock shoot hurry dream divide concert discover"
$MNEMONIC_3 = "fuel obscure melt april direct second usual hair leave hobby beef bacon solid drum used law mercy worry fat super must ritual bring faculty"
$GENESIS_COINS = "10000000000000000000000000stake"
$STAKE_AMOUNT = "1000000000stake"

###############################################################################
###                              Setup Chain                                ###
###############################################################################
Write-Host "`nChain : " $CHAIN_ID -ForegroundColor Green

Remove-Item $CHAIN_DATA -Recurse -Force
$CHAIN_DATA = New-Item -ItemType Directory -Force -Path $CHAIN_DATA

Write-Host "`nWriting chain data at : " $CHAIN_DATA -ForegroundColor Green

Write-Host "`nInitializing chain : " $CHAIN_ID -ForegroundColor Green

& $BINARY_PATH --home $CHAIN_DATA init $CHAIN_ID --chain-id $CHAIN_ID -o

Write-Host "`nAdding genesis accounts" -ForegroundColor Green

$MNEMONIC_1 | & $BINARY_PATH --home $CHAIN_DATA keys add user --recover --keyring-backend test
$MNEMONIC_2 | & $BINARY_PATH --home $CHAIN_DATA keys add user2 --recover --keyring-backend test
$MNEMONIC_3 | & $BINARY_PATH --home $CHAIN_DATA keys add validator --recover --keyring-backend test

$USER = & $BINARY_PATH --home $CHAIN_DATA keys show user --keyring-backend test -a
$USER2 = & $BINARY_PATH --home $CHAIN_DATA keys show user2 --keyring-backend test -a
$VALIDATOR = & $BINARY_PATH --home $CHAIN_DATA keys show validator --keyring-backend test -a

& $BINARY_PATH --home $CHAIN_DATA add-genesis-account $USER $GENESIS_COINS --keyring-backend test
& $BINARY_PATH --home $CHAIN_DATA add-genesis-account $USER2 $GENESIS_COINS --keyring-backend test
& $BINARY_PATH --home $CHAIN_DATA add-genesis-account $VALIDATOR $GENESIS_COINS --keyring-backend test

Write-Host "`nCreating gentx" -ForegroundColor Green

& $BINARY_PATH --home $CHAIN_DATA --chain-id $CHAIN_ID gentx validator $STAKE_AMOUNT --keyring-backend test

Write-Host "`nCollecting gentxs" -ForegroundColor Green

& $BINARY_PATH --home $CHAIN_DATA collect-gentxs

Write-Host "`nValidating genesis file" -ForegroundColor Green

& $BINARY_PATH validate-genesis --home $CHAIN_DATA

# Copy genesis.json to other nodes
$NODE1_CONFIG_DIR = Join-Path -Path $NODE1_DIR -ChildPath "config"
if (!(Test-Path $NODE1_CONFIG_DIR)) {
    New-Item -ItemType Directory -Force -Path $NODE1_CONFIG_DIR
}

Copy-Item -Path "$CHAIN_DATA/config/genesis.json" -Destination "$NODE1_CONFIG_DIR/genesis.json" -Force

}

###############################################################################
###                              Start Chain                                ###
###############################################################################

Write-Host "`nStarting chain : " $CHAIN_ID -ForegroundColor Green

& $BINARY_PATH start --home $CHAIN_DATA --minimum-gas-prices $env:MINIMUM_GAS_PRICES --rpc.laddr "tcp://127.0.0.1:26658" --rpc.pprof_laddr "localhost:6061"
